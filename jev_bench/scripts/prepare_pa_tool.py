#!/usr/bin/env python
"""PA-Tool catalog/generation; preview by default, no GPU/service management."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from jev_bench.advice import json_request
from jev_bench.benchmarks import configure
from jev_bench.config import asset_path, load_config, source_identity
from jev_bench.io import now, write_json
from jev_bench.pa_tool import Projection
from jev_bench.pa_tool_catalog import build_catalog
from jev_bench.pa_tool_generation import CandidateGenerator, generation_cost
from jev_bench.pa_tool_prepare import PROTOCOL, build_mapping, check_protocol, estimate
from jev_bench.pa_tool_reuse import ReuseSource
from jev_bench.runner import run_lock
from jev_bench.services import bypass_local_proxy


def prepare_from_config(args):
    """Extract in the benchmark environment, then start a fresh naming preparation."""
    load_config(args.config)
    asset_path(str(args.catalog_python))
    python = args.catalog_python
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(
            f"PA-Tool catalog interpreter is missing or not executable: {python}"
        )
    script = str(Path(__file__).resolve())
    (PROJECT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pa_prepare_", dir=PROJECT / "tmp") as path:
        catalog = Path(path) / "catalog.json"
        subprocess.run(
            [
                str(python),
                "-B",
                script,
                "catalog",
                "--config",
                str(args.config.resolve()),
                "--output",
                str(catalog),
                "--execute",
            ],
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
        )
        value = json.loads(catalog.read_text())
    # Keep the original prepare command in metadata; no temporary input path survives.
    generate(args, value)


def inside_project(path):
    path = path.resolve()
    if not any(path.is_relative_to(PROJECT / root) for root in ("tmp", "outputs")):
        raise ValueError("PA-Tool artifacts must be inside jev_bench/tmp or outputs")
    return path


def check_generator(settings):
    # Preparation uses only an executor, never simulator/advisor API credentials.
    bypass_local_proxy()
    response = json_request(
        settings["base_url"] + "/models", settings["timeout_seconds"]
    )
    matches = [row for row in response["data"] if row["id"] == settings["model"]]
    if len(matches) != 1:
        raise ValueError("PA-Tool naming service model identity differs")
    row = matches[0]
    if (
        row["root"] != str(Path(settings["model_path"]).resolve(strict=True))
        or row["max_model_len"] != settings["max_model_len"]
    ):
        raise ValueError("PA-Tool naming service path or context limit differs")
    return {key: row[key] for key in ("id", "root", "max_model_len")}


def generate(args, catalog):
    if args.resume and args.reuse_from is not None:
        raise ValueError("--reuse-from cannot be combined with --resume")
    if type(args.concurrency) is not int or args.concurrency < 1:
        raise ValueError("--concurrency must be a positive integer")
    check_protocol(json.loads(args.protocol.read_text()))
    settings = catalog["executor"]
    asset_path(str(args.qwen_model_path))
    asset_path(settings["model_path"])
    if (
        settings["model"] != "qwen3-4b"
        or Path(settings["model_path"]).resolve() != args.qwen_model_path.resolve()
    ):
        raise ValueError("approved PA-Tool naming model is the original Qwen3-4B")
    print(
        json.dumps({**estimate(catalog), "execute": args.execute}, ensure_ascii=False),
        flush=True,
    )
    if not args.execute:
        return
    target = inside_project(args.output_dir)
    code = source_identity(
        PROJECT.parent,
        [
            "jev_bench/src",
            "jev_bench/scripts",
            "jev_bench/configs/pa_tool_prepare.json",
        ],
    )
    if target.is_relative_to(PROJECT / "outputs") and code["dirty_paths"]:
        raise ValueError("formal PA-Tool preparation requires committed code")
    reuse = (
        ReuseSource(args.reuse_from, catalog) if args.reuse_from is not None else None
    )
    service = check_generator(settings)
    additional_settings = (
        ()
        if args.additional_base_url is None
        else ({**settings, "base_url": args.additional_base_url},)
    )
    additional_services = [
        {"base_url": item["base_url"], "health": check_generator(item)}
        for item in additional_settings
    ]
    if not args.resume:
        target.mkdir(parents=True, exist_ok=False)
    elif not (target / "metadata/run_status.json").is_file():
        raise ValueError("PA-Tool resume requires existing run status")
    status_path = target / "metadata/run_status.json"
    ledger = target / "diagnostics/generation.jsonl"
    with run_lock(target) as command:
        if args.resume:
            state = json.loads(status_path.read_text())
            previous = json.loads((target / "catalog.json").read_text())
            if (
                state["status"] not in ("running", "interrupted")
                or previous != catalog
                or state["protocol"] != PROTOCOL
                or state["code"]["trees"] != code["trees"]
                or state["code"]["dirty_paths"]
                or code["dirty_paths"]
                or state["service"] != service
                or state["additional_services"] != additional_services
            ):
                raise ValueError(
                    "PA-Tool resume requires the same committed protocol, catalog, service and code"
                )
        else:
            state = {
                "status": "running",
                "started_at": now(),
                "protocol": PROTOCOL,
                "code": code,
                "service": service,
                "additional_services": additional_services,
                "reuse_source": str(reuse.directory) if reuse is not None else None,
                "reuse_source_status": reuse.source_status
                if reuse is not None
                else None,
            }
            write_json(target / "catalog.json", catalog)
        command.write(shlex.join([sys.executable, *sys.argv]) + "\n")
        command.flush()
        state["status"] = "running"
        state["concurrency"] = args.concurrency
        write_json(status_path, state)
        generator = None
        naming_fallbacks = None
        candidate_outcomes = None
        try:
            generator = CandidateGenerator(
                settings,
                ledger,
                concurrency=args.concurrency,
                reuse=reuse,
                additional_settings=additional_settings,
            )
            mapping = build_mapping(catalog, generator)
            for scope, tools in catalog["scopes"].items():
                Projection(tools, mapping["scopes"][scope])
            failures = mapping["generation_fallbacks"].values()
            naming_fallbacks = {
                "unique_components": len(mapping["generation_fallbacks"]),
                "tools": sum(row["component"] == "tool" for row in failures),
                "parameters": sum(row["component"] == "parameter" for row in failures),
            }
            candidate_outcomes = {}
            for component in ("tool", "parameter"):
                rows = [
                    row
                    for row in mapping["candidate_statistics"].values()
                    if row["component"] == component
                ]
                candidate_outcomes[component] = {
                    "unique_components": len(rows),
                    "valid_candidates_histogram": dict(
                        sorted(Counter(row["valid_candidates"] for row in rows).items())
                    ),
                    "excluded_candidate_slots": sum(
                        len(row["discarded_candidate_slots"]) for row in rows
                    ),
                    "invalid_references": sum(
                        not row["reference_valid"] for row in rows
                    ),
                }
            write_json(target / "mapping.json", mapping)
            state["status"] = "completed"
        except BaseException as exc:
            state["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            pending_error = sys.exc_info()[0] is not None
            try:
                if generator is not None:
                    generator.close()
                write_json(
                    target / "summary.json",
                    {
                        **estimate(catalog),
                        "status": state["status"],
                        "naming_fallbacks": naming_fallbacks,
                        "candidate_outcomes": candidate_outcomes,
                        "preparation_cost": generation_cost(ledger),
                        "new_preparation_cost": generation_cost(ledger, origin="new"),
                        "reused_preparation_cost": generation_cost(
                            ledger, origin="reused"
                        ),
                        "reuse_source": state.get("reuse_source"),
                    },
                )
            except BaseException as exc:
                state["status"] = (
                    "failed" if isinstance(exc, Exception) else "interrupted"
                )
                state["finalization_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if not pending_error:
                    raise
            finally:
                state["updated_at"] = now()
                write_json(status_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    catalog = commands.add_parser(
        "catalog", help="extract public schemas in the benchmark environment"
    )
    catalog.add_argument("--config", type=Path, required=True)
    catalog.add_argument("--executor", default="qwen3_4b")
    catalog.add_argument("--output", type=Path)
    catalog.add_argument(
        "--execute", action="store_true", help="write catalog; no model calls"
    )
    generation = commands.add_parser(
        "generate", help="generate names in llm-efficiency after authorization"
    )
    generation.add_argument("--catalog", type=Path, required=True)
    generation.add_argument(
        "--qwen-model-path",
        type=Path,
        required=True,
        help="absolute deployment path of the approved original Qwen3-4B",
    )
    generation.add_argument(
        "--protocol", type=Path, default=PROJECT / "configs/pa_tool_prepare.json"
    )
    generation.add_argument("--output-dir", type=Path)
    generation.add_argument("--execute", action="store_true")
    recovery = generation.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true")
    recovery.add_argument(
        "--reuse-from",
        type=Path,
        help="import matching successful requests into a new output directory",
    )
    generation.add_argument("--concurrency", type=int, default=1)
    generation.add_argument(
        "--additional-base-url",
        help="one additional endpoint serving the same naming model and context",
    )
    preparation = commands.add_parser(
        "prepare", help="prepare fresh names in llm-efficiency; preview by default"
    )
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument(
        "--catalog-python",
        type=Path,
        required=True,
        help="absolute executable Python path for this benchmark's catalog extraction",
    )
    preparation.add_argument(
        "--qwen-model-path",
        type=Path,
        required=True,
        help="absolute deployment path of the approved original Qwen3-4B",
    )
    preparation.add_argument("--output-dir", type=Path)
    preparation.add_argument("--execute", action="store_true")
    preparation.add_argument("--concurrency", type=int, default=1)
    preparation.add_argument("--additional-base-url")
    preparation.set_defaults(
        protocol=PROJECT / "configs/pa_tool_prepare.json",
        resume=False,
        reuse_from=None,
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.execute and args.output_dir is None:
            parser.error("prepare --execute requires --output-dir")
        prepare_from_config(args)
        return
    if args.stage == "generate":
        if (args.resume or args.reuse_from is not None) and not args.execute:
            parser.error("--resume and --reuse-from require --execute")
        if args.execute and args.output_dir is None:
            parser.error("--execute requires --output-dir")
        generate(args, json.loads(args.catalog.read_text()))
        return
    if args.execute and args.output is None:
        parser.error("catalog --execute requires --output")
    config = load_config(args.config)
    (PROJECT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pa_catalog_", dir=PROJECT / "tmp"
    ) as runtime:
        configure(config, runtime)
        value = build_catalog(config, args.executor)
    print(json.dumps(estimate(value), ensure_ascii=False))
    if args.execute:
        path = inside_project(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
