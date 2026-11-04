"""Dispatch benchmark adapters; live support is defined by each protocol."""

from importlib import import_module

from ..config import BENCHMARKS


def adapter(config):
    name = config["benchmark"]["name"]
    if name not in BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {name}")
    return import_module(f"{__name__}.{name}")


def configure(config, runtime):
    adapter(config).configure(config, runtime)


def load_tasks(config):
    return adapter(config).load_tasks(config)


def run_episode(config, *args, **kwargs):
    from openai import APIStatusError

    try:
        return adapter(config).run_episode(config, *args, **kwargs)
    except APIStatusError as exc:
        # SDK fatal exceptions require response/body keyword arguments when
        # restored and cannot cross spawn. The adapter has saved their original
        # type/body and episode; transport a plain fatal error to the coordinator.
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
