#!/usr/bin/env python
"""Serve calibrated Jev advice on 127.0.0.1 only (mapika environment)."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT / "src"), str(WORKSPACE / "third_party/decider"),
               str(PROJECT.parent / "dataset_tool_inventory/src")]

from jev_train.advisor import JevAdvisor, JevRenderer, make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--temperature", type=Path)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--device")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.render_only:
        if args.tokenizer is None or args.model is not None or args.temperature is not None or args.device is not None:
            parser.error('--render-only requires --tokenizer and does not accept model/temperature/device')
        advisor = JevRenderer(args.tokenizer)
    else:
        if args.model is None or args.temperature is None or args.tokenizer is not None:
            parser.error('prediction requires --model and --temperature; --tokenizer requires --render-only')
        advisor = JevAdvisor(args.model, args.temperature, args.device or 'cuda')
    with make_server(advisor, args.port) as server:
        print(json.dumps({"address": server.server_address, **advisor.describe()}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
