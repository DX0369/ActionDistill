#!/usr/bin/env python3
"""Export tau2 agent tool schemas per domain from the pinned local checkout."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tau2", type=Path, required=True, help="tau2-bench checkout")
    p.add_argument("--domains", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error(f"output already exists: {args.output}")
    os.environ["TAU2_DATA_DIR"] = str(args.tau2 / "data")
    sys.path.insert(0, str(args.tau2 / "src"))
    commit = subprocess.check_output(["git", "-C", str(args.tau2), "rev-parse", "HEAD"], text=True).strip()
    domains = {}
    for domain in args.domains:
        environment = importlib.import_module(f"tau2.domains.{domain}.environment").get_environment()
        domains[domain] = [tool.openai_schema for tool in environment.get_tools()]
    args.output.write_text(json.dumps({"source": str(args.tau2), "commit": commit, "domains": domains},
                                      ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
