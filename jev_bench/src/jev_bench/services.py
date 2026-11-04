"""Read-only service identity checks; this module never starts a service."""

import os
from pathlib import Path

from .advice import json_request, make_advisor
from .config import EXTENDED_BENCHMARKS


def bypass_local_proxy():
    for name in ("NO_PROXY", "no_proxy"):
        hosts = [x for x in os.environ.get(name, "").split(",") if x]
        os.environ[name] = ",".join(dict.fromkeys(hosts + ["127.0.0.1", "localhost"]))


def check_services(config):
    if (
        config["benchmark"]["name"] in EXTENDED_BENCHMARKS
        and config["protocol"]["mode"] == "cpu_only"
    ):
        raise ValueError("CPU-only benchmark: live service protocol is pending")
    if (
        config["benchmark"]["name"] == "toolsandbox"
        and config["protocol"]["mode"] == "formal"
        and not os.environ.get("RAPID_API_KEY", "").strip()
    ):
        raise ValueError("formal ToolSandbox requires a nonempty RAPID_API_KEY")
    bypass_local_proxy()
    observed = {"executors": {}, "advisors": {}}
    if config["benchmark"]["name"] in ("tau2", "toolsandbox"):
        user = config["user_simulator"]
        if not os.environ.get(user["api_key_env"], "").strip():
            raise ValueError(
                "user simulator API key environment variable is missing or empty"
            )
        # Do not spend a simulated-user request during preflight or persist credentials.
        observed["user_simulator"] = {
            "base_url": user["base_url"],
            "model": user["model"],
            "remote_checked": False,
        }
    if config["benchmark"]["name"] == "stabletoolbench":
        backend = config["protocol"]["backend"]
        participants = {
            "tool_backend": backend,
            "tool_simulator": backend["simulator"],
            "judge": config["protocol"]["judge"],
        }
        for role, settings in participants.items():
            field = "toolbench_key_env" if role == "tool_backend" else "api_key_env"
            if not os.environ.get(settings[field], "").strip():
                raise ValueError(
                    f"{role}: credential environment variable is missing or empty"
                )
            observed[role] = {"remote_checked": False}
    if config["benchmark"]["name"] == "apibank":
        from .benchmarks.apibank_group_scoring import scoring_python

        scoring_python()
    for name in dict.fromkeys(c["executor"] for c in config["conditions"].values()):
        settings = config["executors"][name]
        response = json_request(
            settings["base_url"] + "/models", settings["timeout_seconds"]
        )
        matching = [
            model for model in response["data"] if model["id"] == settings["model"]
        ]
        if len(matching) != 1:
            raise ValueError(
                f"executor {name}: expected exactly one matching served model"
            )
        model = matching[0]
        if model["root"] != str(Path(settings["model_path"]).resolve(strict=True)):
            raise ValueError(f"executor {name}: model path differs")
        if (
            type(model["max_model_len"]) is not int
            or model["max_model_len"] != settings["max_model_len"]
        ):
            raise ValueError(f"executor {name}: context limit differs")
        observed["executors"][name] = {
            key: model[key] for key in ("id", "root", "max_model_len")
        }
    for name in dict.fromkeys(
        c["advisor"] for c in config["conditions"].values() if "advisor" in c
    ):
        client = make_advisor(
            config["advisors"][name], lambda *_: None, advisors=config["advisors"]
        )
        try:
            observed["advisors"][name] = client.health()
        finally:
            client.close()
    return observed
