"""One serial chain per executor endpoint; the parent owns the final index."""

import fcntl
import hashlib
import json
import multiprocessing
import random
import shutil
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from .attempt import recover_interrupted
from .benchmarks import configure, run_episode
from .config import (
    FORMAL_TASK_COUNTS,
    PROJECT_ROOT,
    STABLETOOLBENCH_CATEGORIES,
    fields,
    number,
)
from .io import append_json, now, read_jsonl, write_json


class ServiceUnavailable(RuntimeError):
    """A named service failure leaves the planned episode unresolved."""


@contextmanager
def run_lock(run_dir):
    """Lock the existing command record, avoiding a second coordinator and extra lock artifacts."""
    path = Path(run_dir) / "metadata/command.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another coordinator owns this run") from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def key(item):
    return item["condition"], item["category"], item["task_id"], item["trial"]


def resolved(row):
    """Scoring completion is independent of boolean versus continuous outcomes."""
    return row["official_scored"] or row["success"] is not None


def make_plan(config, tasks):
    formal = config.get("protocol", {}).get("mode") == "formal"
    benchmark = config["benchmark"]["name"]
    if formal:
        if (
            dict(Counter(category for category, _ in tasks))
            != FORMAL_TASK_COUNTS[benchmark]
        ):
            raise ValueError("formal input coverage differs from the frozen benchmark")
        if benchmark == "toolsandbox":
            families = Counter(task["family_id"] for task in tasks.values())
            variants = {(task["family_id"], task["variant"]) for task in tasks.values()}
            if (
                len(families) != 129
                or set(families.values()) != {8}
                or len(variants) != 1032
            ):
                raise ValueError(
                    "formal ToolSandbox requires 129 complete eight-variant families"
                )
    metadata = {}
    if formal and benchmark in ("apibank", "toolsandbox"):
        names = (
            ("source_id",)
            if benchmark == "apibank"
            else ("family_id", "variant", "categories")
        )
        metadata = {
            identity: {name: task[name] for name in names}
            for identity, task in tasks.items()
        }
    identities = sorted(tasks)
    random.Random(config["seed"]).shuffle(identities)
    return [
        {
            "condition": condition,
            "task_id": task_id,
            "category": category,
            "trial": trial,
            "seed": config["seed"] + index * config["trials"] + trial,
            **metadata.get((category, task_id), {}),
        }
        for index, (category, task_id) in enumerate(identities)
        for trial in range(config["trials"])
        for condition in config["conditions"]
    ]


def make_chains(config, plan):
    chains = {}
    for item in plan:
        executor = config["conditions"][item["condition"]]["executor"]
        url = config["executors"][executor]["base_url"]
        chains.setdefault(url, []).append(item)
    return chains


def validate_attempts(rows, plan):
    planned = {key(item): item for item in plan}
    if len(planned) != len(plan):
        raise ValueError("duplicate planned identity")
    latest = {}
    for row in rows:
        identity = key(row)
        if identity not in planned or row["category"] != planned[identity]["category"]:
            raise ValueError("attempt does not match the frozen plan")
        for field in ("source_id", "family_id", "variant", "categories"):
            if field in planned[identity] and row[field] != planned[identity][field]:
                raise ValueError("attempt metadata differs from the frozen plan")
        previous = latest.get(identity)
        expected = 1 if previous is None else previous["attempt"] + 1
        if type(row["attempt"]) is not int or row["attempt"] != expected:
            raise ValueError("duplicate or non-contiguous attempt number")
        if previous is not None and (
            resolved(previous) or previous["failure_kind"] == "pending_scoring"
        ):
            raise ValueError(
                "a scored or protocol-counted model outcome must never be rerun"
            )
        if type(row["official_scored"]) is not bool or (
            row["success"] is not None and type(row["success"]) is not bool
        ):
            raise ValueError(
                "success and official_scored must use canonical boolean values"
            )
        if row["official_scored"]:
            expected = (
                "similarity"
                if row["category"] == "scenarios"
                else "rouge_l_f"
                if row["category"] in {f"level_{level}_response" for level in (1, 2, 3)}
                else "solvable_pass_rate"
                if row["category"] in STABLETOOLBENCH_CATEGORIES
                else None
            )
            if expected is not None and row["success"] is not None:
                raise ValueError(
                    "continuous official scores must not become boolean success"
                )
            if row["success"] is None:
                score = row["official_score"]
                fields(score, ("metric", "value"))
                if expected is None or score["metric"] != expected:
                    raise ValueError(
                        "continuous score metric differs from its benchmark category"
                    )
                number(score["value"], maximum=1)
                number(row[expected], maximum=1)
                if row[expected] != score["value"]:
                    raise ValueError("official score differs from its canonical value")
                if expected == "similarity":
                    from .benchmarks.toolsandbox_scores import validate_score

                    validate_score(row)
                if row["failure_kind"] is not None:
                    raise ValueError(
                        "completed continuous scoring cannot have a failure"
                    )
            elif row["failure_kind"] != (
                None if row["success"] else "model_task_failure"
            ):
                raise ValueError(
                    "official scores and failure classification contradict each other"
                )
        elif row["success"] is None:
            if row["failure_kind"] not in {
                "service_error",
                "user_simulator_error",
                "scoring_error",
                "unexpected_error",
                "interrupted",
                "advisor_error",
                "pending_scoring",
            }:
                raise ValueError("unresolved outcomes require a named failure")
        elif row["success"] is not False or row["failure_kind"] not in {
            "executor_context_limit",
            "executor_response_error",
        }:
            raise ValueError(
                "unscored outcomes require a named protocol-counted executor failure"
            )
        latest[identity] = row
    return latest


def pending_plan(plan, rows):
    latest = validate_attempts(rows, plan)
    pending = []
    for item in plan:
        old = latest.get(key(item))
        if (
            old is not None
            and not resolved(old)
            and old["failure_kind"]
            not in (
                "service_error",
                "user_simulator_error",
                "interrupted",
                "pending_scoring",
            )
        ):
            raise ValueError("fatal attempts require diagnosis and cannot be resumed")
        if old is None or (
            not resolved(old) and old["failure_kind"] != "pending_scoring"
        ):
            pending.append(
                {**item, "attempt": 1 if old is None else old["attempt"] + 1}
            )
    return pending


def directory_for(run_dir, item):
    return (
        Path(run_dir)
        / "episodes"
        / item["condition"]
        / item["category"]
        / hashlib.sha256(item["task_id"].encode("utf-8")).hexdigest()
        / f"trial_{item['trial']}"
        / f"attempt_{item['attempt']}"
    )


def reconcile(run_dir, plan, *, recover=False):
    """Index finished results; explicit resume may record a dead inference attempt."""
    rows = read_jsonl(Path(run_dir) / "episodes.jsonl")
    indexed = {(key(row), row["attempt"]) for row in rows}
    latest = validate_attempts(rows, plan)
    for item in plan:
        previous = latest.get(key(item))
        if previous is not None and previous["failure_kind"] == "interrupted":
            # A persisted soft-interrupt row must obey the same no-resampling
            # boundary as an attempt recovered after a killed worker.
            recover_interrupted(
                directory_for(run_dir, previous),
                {**item, "attempt": previous["attempt"]},
            )
        attempt = 1 if previous is None else previous["attempt"] + 1
        directory = directory_for(run_dir, {**item, "attempt": attempt})
        if not directory.exists():
            continue
        result = directory / "episode.json"
        if not result.exists():
            if not recover:
                raise ValueError(
                    f"incomplete attempt requires explicit resume/diagnosis: {directory}"
                )
            row = recover_interrupted(directory, {**item, "attempt": attempt})
        else:
            row = json.loads(result.read_text())
            if row["failure_kind"] == "interrupted":
                row = recover_interrupted(directory, {**item, "attempt": attempt})
        if key(row) != key(item) or row["attempt"] != attempt:
            raise ValueError(
                "recovered result identity differs from its attempt directory"
            )
        if (key(row), row["attempt"]) in indexed:
            raise ValueError("duplicate recovered attempt")
        validate_attempts(rows + [row], plan)
        append_json(Path(run_dir) / "episodes.jsonl", row)
        rows.append(row)
    return rows


def execute(config, plan, tasks, answers, run_dir, state):
    rows = reconcile(run_dir, plan, recover=True)
    pending = pending_plan(plan, rows)
    chains = make_chains(config, pending)
    runtime = (
        PROJECT_ROOT
        / "tmp"
        / ("runtime_" + Path(run_dir).parent.name + "_" + Path(run_dir).name)
    )
    context = multiprocessing.get_context("spawn")
    pools, futures, queues = [], {}, {}
    failure = None
    scoring_attempts = []
    only_scoring_failures = True
    raising_worker_failure = False
    state_path = Path(run_dir) / "metadata/run_status.json"
    state.update(status="running", last_started_at=now(), active=[], attempts=len(rows))
    write_json(state_path, state)

    def submit(pool, queue):
        item = queue.pop(0)
        directory = directory_for(run_dir, item)
        kwargs = {}
        if "pa_tool_mapping" in config["conditions"][item["condition"]]:
            kwargs["pa_tool_identity"] = state["pa_tool_assets"][item["condition"]]
        future = pool.submit(
            run_episode,
            config,
            item,
            tasks[(item["category"], item["task_id"])],
            answers[(item["category"], item["task_id"])],
            directory,
            **kwargs,
        )
        futures[future] = (pool, item)
        state["active"] = [active for _, active in futures.values()]
        write_json(state_path, state)

    try:
        for index, items in enumerate(chains.values()):
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=configure,
                initargs=(config, runtime / str(index)),
            )
            pools.append(pool)
            queues[pool] = list(items)
            submit(pool, queues[pool])
        while futures:
            ready, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in ready:
                pool, item = futures.pop(future)
                received_result = False
                try:
                    row = future.result()
                    received_result = True
                    if key(row) != key(item) or row["attempt"] != item["attempt"]:
                        raise ValueError(
                            "worker result identity differs from the submitted episode"
                        )
                    validate_attempts(rows + [row], plan)
                    append_json(Path(run_dir) / "episodes.jsonl", row)
                    rows.append(row)
                    print(
                        json.dumps(
                            {
                                **item,
                                "success": row["success"],
                                "failure_kind": row["failure_kind"],
                            }
                        ),
                        flush=True,
                    )
                    if (
                        not resolved(row)
                        and row["failure_kind"] != "pending_scoring"
                        and failure is None
                    ):
                        failure = ServiceUnavailable(
                            "repair the service, then explicitly resume"
                        )
                except BaseException as exc:
                    failure = exc
                    fatal = None
                    result_path = directory_for(run_dir, item) / "episode.json"
                    if result_path.exists() and not any(
                        key(row) == key(item) and row["attempt"] == item["attempt"]
                        for row in rows
                    ):
                        fatal = json.loads(result_path.read_text())
                        if (
                            key(fatal) != key(item)
                            or fatal["attempt"] != item["attempt"]
                        ):
                            raise ValueError(
                                "fatal result identity differs from the submitted episode"
                            ) from exc
                        validate_attempts(rows + [fatal], plan)
                        append_json(Path(run_dir) / "episodes.jsonl", fatal)
                        rows.append(fatal)
                    if (
                        not received_result
                        and fatal is not None
                        and fatal["failure_kind"] == "scoring_error"
                    ):
                        scoring_attempts.append(item)
                    else:
                        only_scoring_failures = False
                state.update(
                    attempts=len(rows),
                    active=[active for _, active in futures.values()],
                )
                write_json(state_path, state)
                if failure is None and queues[pool]:
                    submit(pool, queues[pool])
        if failure is None and not pending_plan(plan, rows):
            state.update(status="running", phase="episodes_complete")
        else:
            state["status"] = "partial"
        if failure is not None:
            state["error"] = {"type": type(failure).__name__, "message": str(failure)}
            if not isinstance(failure, ServiceUnavailable):
                state["status"] = (
                    "interrupted"
                    if isinstance(failure, KeyboardInterrupt)
                    else "failed"
                )
                raising_worker_failure = True
                raise failure
    except BaseException as exc:
        state.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        if raising_worker_failure and only_scoring_failures and scoring_attempts:
            state["error"]["scoring_attempts"] = scoring_attempts
        raise
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
        state.update(ended_at=now(), active=[])
        write_json(state_path, state)
        if runtime.exists():
            shutil.rmtree(runtime)
    return rows
