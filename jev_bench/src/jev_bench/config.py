"""Validate explicit benchmark protocols without encoding condition names."""

import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
CATEGORIES = tuple(
    "multi_turn_" + name for name in ("base", "miss_func", "miss_param", "long_context")
)
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
TAU2_COMMIT = "3d82db8f3579efe29609ab9daff058a9b7cb187c"
TOOLSANDBOX_COMMIT = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
APIBANK_COMMIT = "faf754671f5b850e1b4d1ff870db37a6a2480df6"
TOOLHOP_REVISION = "b439d7279af359fda46e8117ae4f0245b75f5c6b"
STABLETOOLBENCH_COMMIT = "aa4ed9f4737ad98bd706663f01d63623c3427812"
STABLETOOLBENCH_CATEGORIES = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)
EXTENDED_BENCHMARKS = ("toolsandbox", "apibank", "toolhop", "stabletoolbench")
APIBANK_CATEGORIES = tuple(
    f"level_{level}_{task}" for level in (1, 2, 3) for task in ("api", "response")
)
TAU2_DOMAINS = ("retail", "airline", "telecom")
BENCHMARKS = {
    "bfcl": (
        "third_party/gorilla/berkeley-function-call-leaderboard",
        BFCL_COMMIT,
        CATEGORIES,
    ),
    "tau2": ("third_party/tau2-bench-verified", TAU2_COMMIT, TAU2_DOMAINS),
    "toolsandbox": ("third_party/ToolSandbox", TOOLSANDBOX_COMMIT, ("scenarios",)),
    "apibank": ("third_party/DAMO-ConvAI/api-bank", APIBANK_COMMIT, APIBANK_CATEGORIES),
    "toolhop": ("third_party/ToolHop", TOOLHOP_REVISION, ("tasks",)),
    "stabletoolbench": (
        "third_party/StableToolBench",
        STABLETOOLBENCH_COMMIT,
        STABLETOOLBENCH_CATEGORIES,
    ),
}
FORMAL_TASK_COUNTS = {
    "toolhop": {"tasks": 995},
    "apibank": dict(
        zip(APIBANK_CATEGORIES, (399, 375, 135, 103, 245, 50), strict=True)
    ),
    "toolsandbox": {"scenarios": 1032},
    "stabletoolbench": dict(
        zip(STABLETOOLBENCH_CATEGORIES, (163, 153, 158, 106, 124, 61), strict=True)
    ),
}
RUNTIME_PATHS = {
    "bfcl": ("bfcl_eval",),
    "tau2": ("src/tau2", "data"),
    "toolsandbox": ("tool_sandbox",),
    "apibank": (
        "api_call_extraction.py",
        "tool_manager.py",
        "evaluator_by_json.py",
        "lv3_evaluator.py",
        "utils.py",
        "apis",
        "lv3_apis",
        "init_database",
        "lv1-lv2-samples",
    ),
    "stabletoolbench": (
        "server/main.py",
        "server/utils.py",
        "toolbench/inference",
        "toolbench/tooleval",
        "solvable_queries/test_instruction",
        "solvable_queries/test_query_ids",
    ),
}


def git_dirty(repository, paths):
    entries = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *paths,
        ],
        text=True,
    ).splitlines()
    # Ignore only untracked interpreter caches, never modified tracked inputs.
    return [
        entry
        for entry in entries
        if not (
            entry.startswith("?? ")
            and "/__pycache__/" in entry
            and entry.endswith(".pyc")
        )
    ]


def source_identity(repository, paths):
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    trees, dirty = {}, git_dirty(repository, paths)
    for path in paths:
        entry = subprocess.check_output(
            ["git", "-C", str(repository), "ls-tree", "HEAD", "--", path], text=True
        ).strip()
        trees[path] = entry.split()[2] if entry else None
        if not entry:
            dirty.append(f"untracked runtime input: {path}")
    return {"commit": commit, "trees": trees, "dirty_paths": dirty}


TAU2_PROTOCOL = {
    "split": "base",
    "max_steps": 200,
    "max_errors": 10,
    "evaluation_type": "ALL",
    "solo_mode": False,
    "validate_communication": False,
}
GATES = {"respond_prob_ge": 0.1, "top1_prob_lt": 0.3}
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}


def fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    missing, extra = (
        set(required) - value.keys(),
        value.keys() - set(required) - set(optional),
    )
    if missing or extra:
        raise ValueError(
            f"missing fields {sorted(missing)}; unexpected fields {sorted(extra)}"
        )


def number(value, *, minimum=0, maximum=math.inf):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"expected a finite number in [{minimum}, {maximum}], got {value!r}"
        )


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected an integer >= {minimum}, got {value!r}")


def advisor_type(config, condition):
    """None for none/neutral conditions; validated advisors carry an explicit type."""
    policy = config["conditions"][condition]
    return (
        config["advisors"][policy["advisor"]]["type"] if "advisor" in policy else None
    )


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError(f"invalid identifier: {value!r}")


def local_url(value, path):
    if not isinstance(value, str):
        raise TypeError("service URL must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path != path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"expected http://127.0.0.1:<port>{path}")


def asset_path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("asset paths must be explicit absolute paths")


def validate_user(user):
    fields(
        user,
        (
            "base_url",
            "model",
            "api_key_env",
            "temperature",
            "max_tokens",
            "thinking",
            "timeout_seconds",
        ),
    )
    if (
        user["base_url"] != "https://api.deepseek.com"
        or user["model"] != "deepseek-flash"
    ):
        raise ValueError(
            "the approved simulator requires official HTTPS DeepSeek and deepseek-flash"
        )
    for field in ("model", "api_key_env"):
        identifier(user[field])
    number(user["temperature"], maximum=0)
    integer(user["max_tokens"], 1)
    number(user["timeout_seconds"], minimum=0.001)
    if user["thinking"] != "disabled":
        raise ValueError("the approved simulated user requires thinking=disabled")


def validate(config):
    fields(
        config,
        (
            "benchmark",
            "output_root",
            "task_ids",
            "trials",
            "seed",
            "executors",
            "advisors",
            "conditions",
            "comparisons",
        ),
        ("protocol", "user_simulator"),
    )
    fields(config["benchmark"], ("name", "path", "commit"))
    name = config["benchmark"]["name"]
    if name not in BENCHMARKS:
        raise ValueError("unsupported benchmark")
    path, commit, categories = BENCHMARKS[name]
    if config["benchmark"] != {"name": name, "path": path, "commit": commit}:
        raise ValueError("benchmark source must use the pinned local version")
    if name == "tau2":
        fields(config["protocol"], TAU2_PROTOCOL)
        if any(
            type(config["protocol"][k]) is not type(v) or config["protocol"][k] != v
            for k, v in TAU2_PROTOCOL.items()
        ):
            raise ValueError("use the approved tau2 base/manual/dual-control protocol")
        validate_user(config["user_simulator"])
    elif name == "toolsandbox":
        fields(config["protocol"], ("mode", "scenario_seed"))
        if (
            config["protocol"]["mode"] not in ("cpu_only", "smoke", "formal")
            or type(config["protocol"]["scenario_seed"]) is not int
            or config["protocol"]["scenario_seed"] != 42
            or (
                config["protocol"]["mode"] == "cpu_only"
                and config["user_simulator"] is not None
            )
        ):
            raise ValueError(
                "ToolSandbox requires catalog seed 42 and a supported mode; "
                "CPU preview requires user_simulator=null"
            )
        if config["protocol"]["mode"] != "cpu_only":
            validate_user(config["user_simulator"])
        if config["protocol"]["mode"] == "smoke":
            allowed = {
                "send_message_with_contact_content_cellular_off_multiple_user_turn",
                "remove_contact_by_phone_no_search_contacts_insufficient_information",
                "remove_contact_with_id",
                "wifi_off_3_distraction_tools_tool_description_scrambled",
                "wifi_off_3_distraction_tools_arg_description_scrambled",
            }
            if (
                not isinstance(config["task_ids"]["scenarios"], list)
                or not set(config["task_ids"]["scenarios"]) <= allowed
            ):
                raise ValueError(
                    "ToolSandbox smoke is limited to the approved local scenarios"
                )
    elif name == "apibank":
        fields(config["protocol"], ("mode", "data_root", "retriever_model_path"))
        if (
            config["protocol"]["mode"] not in ("cpu_only", "smoke", "formal")
            or "user_simulator" in config
        ):
            raise ValueError("API-Bank requires a supported mode and no simulated user")
        for field in ("data_root", "retriever_model_path"):
            asset_path(config["protocol"][field])
    elif name == "toolhop":
        protocol = config["protocol"]
        fields(protocol, ("mode",), ("scenario", "max_turns"))
        if "user_simulator" in config:
            raise ValueError("ToolHop does not use a simulated user")
        if protocol["mode"] in ("smoke", "formal"):
            fields(protocol, ("mode", "scenario", "max_turns"))
            if (
                protocol["scenario"] != "Free"
                or type(protocol["max_turns"]) is not int
                or protocol["max_turns"] != 9
            ):
                raise ValueError(
                    "ToolHop requires the approved Free/max_turns=9 protocol"
                )
        elif protocol != {"mode": "cpu_only"}:
            raise ValueError("unknown ToolHop execution mode")
    elif name == "stabletoolbench":
        protocol = config["protocol"]
        fields(
            protocol,
            ("mode", "cache_archive"),
            (
                "backend",
                "judge",
                "judge_repeats",
                "single_chain_max_step",
                "max_observation_length",
            ),
        )
        if protocol["mode"] == "cpu_only":
            fields(protocol, ("mode", "cache_archive"))
        elif protocol["mode"] in ("smoke", "formal"):
            fields(
                protocol,
                (
                    "mode",
                    "cache_archive",
                    "backend",
                    "judge",
                    "judge_repeats",
                    "single_chain_max_step",
                    "max_observation_length",
                ),
            )
            for field, value in (
                ("judge_repeats", 3),
                ("single_chain_max_step", 12),
                ("max_observation_length", 1024),
            ):
                if type(protocol[field]) is not int or protocol[field] != value:
                    raise ValueError(f"StableToolBench requires {field}={value}")
            backend = protocol["backend"]
            fields(
                backend,
                ("toolbench_url", "toolbench_key_env", "timeout_seconds", "simulator"),
            )
            if backend["toolbench_url"] != "http://8.130.32.149:8080/rapidapi":
                raise ValueError("use the pinned ToolBench endpoint")
            identifier(backend["toolbench_key_env"])
            number(backend["timeout_seconds"], minimum=0.001)
            for settings, temperature, tokens in (
                (backend["simulator"], 0, 1024),
                (protocol["judge"], 0.2, 1000),
            ):
                fields(
                    settings,
                    (
                        "base_url",
                        "model",
                        "api_key_env",
                        "temperature",
                        "max_tokens",
                        "thinking",
                        "timeout_seconds",
                    ),
                )
                if (
                    settings["base_url"] != "https://api.deepseek.com"
                    or settings["model"] != "deepseek-flash"
                    or settings["thinking"] != "disabled"
                    or settings["temperature"] != temperature
                    or type(settings["max_tokens"]) is not int
                    or settings["max_tokens"] != tokens
                ):
                    raise ValueError(
                        "invalid approved StableToolBench provider protocol"
                    )
                identifier(settings["api_key_env"])
                number(settings["timeout_seconds"], minimum=0.001)
        else:
            raise ValueError("unsupported StableToolBench mode")
        if "user_simulator" in config:
            raise ValueError("StableToolBench does not use a user simulator")
        asset_path(protocol["cache_archive"])
    elif "protocol" in config or "user_simulator" in config:
        raise ValueError("BFCL must not configure a simulated user or tau2 protocol")
    if name in EXTENDED_BENCHMARKS:
        formal = config["protocol"]["mode"] == "formal"
        if formal != (config["output_root"] == "outputs"):
            raise ValueError(
                "formal mode requires outputs; engineering modes require tmp"
            )
        if not formal and config["comparisons"]:
            raise ValueError("engineering modes do not permit inferential comparisons")
    if (
        name in EXTENDED_BENCHMARKS
        and config["protocol"]["mode"] == "smoke"
        and (
            config["trials"] != 1
            or config["comparisons"]
            or any(ids == "all" for ids in config["task_ids"].values())
        )
    ):
        raise ValueError(
            "engineering smoke requires explicit subsets, one trial and no comparisons"
        )
    if config["output_root"] not in ("tmp", "outputs"):
        raise ValueError("output_root must be tmp or outputs")
    integer(config["trials"], 1)
    if config["output_root"] == "outputs" and config["trials"] != 1:
        raise ValueError(
            "the approved formal protocol requires one trial per task and condition"
        )
    integer(config["seed"])
    fields(config["task_ids"], categories)
    partial_smoke = (
        name in ("bfcl", "tau2", "apibank")
        and config["output_root"] == "tmp"
        and config["trials"] == 1
        and not config["comparisons"]
    )
    for category, ids in config["task_ids"].items():
        if ids == "all":
            continue
        if (
            not isinstance(ids, list)
            or (not ids and not partial_smoke)
            or any(not isinstance(x, str) for x in ids)
        ):
            raise ValueError("task_ids must be all or a nonempty list of canonical IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("task IDs must be unique and belong to their category")
        for task_id in ids:
            if not task_id:
                raise ValueError("task ID must not be empty")
            if name == "bfcl" and not re.fullmatch(category + r"_\d+", task_id):
                raise ValueError("BFCL task ID must belong to its category")
    if not any(config["task_ids"].values()):
        raise ValueError("at least one benchmark task must be selected")
    if config["output_root"] == "outputs" and any(
        x != "all" for x in config["task_ids"].values()
    ):
        raise ValueError(
            "formal outputs require the complete benchmark; subsets belong in tmp"
        )
    if not isinstance(config["executors"], dict) or not config["executors"]:
        raise ValueError("executors must be a nonempty object")
    endpoints = {}
    for name, executor in config["executors"].items():
        identifier(name)
        fields(
            executor,
            (
                "base_url",
                "model",
                "model_path",
                "timeout_seconds",
                "max_tokens",
                "max_model_len",
                "yarn",
                "thinking",
            ),
        )
        local_url(executor["base_url"], "/v1")
        identifier(executor["model"])
        asset_path(executor["model_path"])
        number(executor["timeout_seconds"], minimum=0.001)
        if (
            type(executor["max_model_len"]) is not int
            or executor["max_model_len"] != 131072
        ):
            raise ValueError("all conditions require 131072 total context")
        if type(executor["max_tokens"]) is not int or executor["max_tokens"] != 8192:
            raise ValueError("the approved output budget is 8192")
        if executor["yarn"] != YARN:
            raise ValueError("the approved YaRN parameters are required")
        fields(executor["thinking"], ("temperature", "top_p", "top_k", "min_p"))
        for key in ("temperature", "top_p", "min_p"):
            number(executor["thinking"][key], maximum=2 if key == "temperature" else 1)
        integer(executor["thinking"]["top_k"], 1)
        if executor["thinking"] != THINKING:
            raise ValueError(
                "all benchmarks require the approved Qwen thinking sampling"
            )
        url = executor["base_url"]
        if url in endpoints and endpoints[url] != executor:
            raise ValueError("one endpoint cannot declare different executor settings")
        endpoints[url] = executor
    if not isinstance(config["advisors"], dict):
        raise TypeError("advisors must be an object")
    for name, advisor in config["advisors"].items():
        identifier(name)
        common = ("type", "base_url", "timeout_seconds")
        kind = advisor["type"]
        if kind == "jev_local":
            fields(advisor, (*common, "model", "temperature_file"))
            local_url(advisor["base_url"], "")
            asset_path(advisor["model"])
            asset_path(advisor["temperature_file"])
        elif kind == 'jev_render':
            fields(advisor, (*common, 'tokenizer'))
            local_url(advisor['base_url'], '')
            asset_path(advisor['tokenizer'])
        elif kind in ("typesafe", "deepseek"):
            if (
                config["benchmark"]["name"] not in ("bfcl", "tau2")
                and config["protocol"]["mode"] == "cpu_only"
            ):
                raise ValueError("remote advisors require a live protocol")
            extra = (
                ("render_advisor",)
                if kind == "typesafe"
                else ("max_tokens", "reasoning_effort")
            )
            fields(advisor, (*common, "model", "api_key_env", *extra))
            endpoint, model = (
                ("https://api.typesafe.ai/v1", "jev-1.14.0")
                if kind == "typesafe"
                else ("https://api.deepseek.com", "deepseek-flash")
            )
            if advisor["base_url"] != endpoint or advisor["model"] != model:
                raise ValueError(f"{kind} requires {endpoint} and model {model}")
            if not isinstance(advisor["api_key_env"], str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", advisor["api_key_env"]
            ):
                raise ValueError("api_key_env must name an environment variable")
            if kind == "typesafe":
                renderer = advisor["render_advisor"]
                identifier(renderer)
                if renderer not in config["advisors"] or (
                    config["advisors"][renderer]["type"] not in ('jev_local', 'jev_render')
                ):
                    raise ValueError("render_advisor must name a Jev predictor or tokenizer renderer")
            elif (
                type(advisor["max_tokens"]) is not int
                or advisor["max_tokens"] != 16384
                or advisor["reasoning_effort"] != "high"
            ):
                raise ValueError(
                    "DeepSeek advisor requires 16384 output tokens and high effort"
                )
        else:
            raise ValueError("unsupported advisor type")
        number(advisor["timeout_seconds"], minimum=0.001)
    if not isinstance(config["conditions"], dict) or not config["conditions"]:
        raise ValueError("conditions must be a nonempty object")
    for name, condition in config["conditions"].items():
        identifier(name)
        fields(
            condition, ("executor", "advice"), ("advisor", "gate", "pa_tool_mapping", "hint_display")
        )
        if condition["executor"] not in config["executors"]:
            raise ValueError("unknown executor")
        mode = condition["advice"]
        if "pa_tool_mapping" in condition:
            asset_path(condition["pa_tool_mapping"])
            if mode != "none":
                raise ValueError("PA-Tool requires advice=none")
        if mode not in ("none", "neutral", "prefer", "gate"):
            raise ValueError("unsupported advice policy")
        if mode in ("prefer", "gate"):
            if condition.get("advisor") not in config["advisors"]:
                raise ValueError("advice requires a configured advisor")
            if config['advisors'][condition['advisor']]['type'] == 'jev_render':
                raise ValueError('a tokenizer renderer cannot be an action advisor')
        elif "advisor" in condition:
            raise ValueError("none/neutral must not configure an advisor")
        if mode == "gate":
            if config["advisors"][condition["advisor"]]["type"] == "deepseek":
                raise ValueError(
                    "DeepSeek has no candidate probabilities; gate is unsupported"
                )
            fields(condition["gate"], ("rule", "threshold", "loop_guard"))
            gate = condition["gate"]
            if (
                gate["rule"] not in GATES
                or gate["threshold"] != GATES[gate["rule"]]
                or gate["loop_guard"] is not True
            ):
                raise ValueError("use the unchanged G1 or G2 rule and loop_guard=true")
        elif "gate" in condition:
            raise ValueError("gate parameters require gate advice")
        if 'hint_display' in condition:
            if mode not in ('prefer', 'gate') or config['advisors'][condition['advisor']]['type'] != 'jev_local':
                raise ValueError('adaptive display requires a local Jev action advisor')
            display = condition['hint_display']
            fields(display, ('mode', 'threshold'))
            if (display['mode'] != 'adaptive_top3' or type(display['threshold']) not in (int, float)
                    or display['threshold'] != 0.5):
                raise ValueError('adaptive_top3 requires the explicit threshold 0.5')
    if not isinstance(config["comparisons"], list):
        raise TypeError("comparisons must be a list of [treatment, reference]")
    seen = set()
    for pair in config["comparisons"]:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] == pair[1]
            or any(x not in config["conditions"] for x in pair)
        ):
            raise ValueError("comparison must name two different configured conditions")
        if tuple(pair) in seen:
            raise ValueError("duplicate comparison")
        seen.add(tuple(pair))
    return config


def load_config(path):
    config = validate(json.loads(Path(path).read_text()))
    return check_checkout(config)


def check_checkout(config):
    """Check only the selected official runtime, including local source edits."""
    if config["benchmark"]["name"] == "toolhop":
        # hf download snapshots carry the source revision without a Git checkout.
        source = WORKSPACE_ROOT / config["benchmark"]["path"]
        for filename in ("data/ToolHop.json", "code/evaluation_closed.py"):
            metadata = source / ".cache/huggingface/download" / (filename + ".metadata")
            with metadata.open() as stream:
                revision = stream.readline().strip()
            if revision != TOOLHOP_REVISION:
                raise ValueError("ToolHop snapshot differs from the pinned revision")
        return config
    actual = subprocess.check_output(
        [
            "git",
            "-C",
            str(WORKSPACE_ROOT / config["benchmark"]["path"]),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if actual != config["benchmark"]["commit"]:
        raise ValueError("local benchmark checkout differs from the pinned commit")
    dirty = git_dirty(
        WORKSPACE_ROOT / config["benchmark"]["path"],
        RUNTIME_PATHS[config["benchmark"]["name"]],
    )
    if dirty:
        raise ValueError(f"local benchmark runtime has uncommitted changes: {dirty}")
    return config
