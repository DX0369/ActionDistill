"""API-Bank isolated official scoring worker.

One process scores one arm/trial in the existing jev-apibank environment.
Only JSON is written to stdout; upstream diagnostics remain on stderr.
"""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    with redirect_stdout(sys.stderr):
        from jev_bench.benchmarks.apibank import configure
        from jev_bench.benchmarks.apibank_scoring import score_batch
        from jev_bench.config import fields

        payload = json.load(sys.stdin)
        fields(payload, ("config", "predictions", "terminal_failures"))
        if not isinstance(payload["predictions"], list):
            raise TypeError("prediction transport must be an ordered record list")
        predictions = {}
        for row in payload["predictions"]:
            fields(row, ("category", "task_id", "prediction"))
            if not isinstance(row["category"], str) or not isinstance(
                row["task_id"], str
            ):
                raise TypeError("prediction identities must be canonical strings")
            identity = row["category"], row["task_id"]
            if identity in predictions:
                raise ValueError("duplicate prediction identity in score transport")
            predictions[identity] = row["prediction"]
        configure(payload["config"], None)
        if not isinstance(payload["terminal_failures"], list):
            raise TypeError("terminal failures must be an explicit record list")
        terminal_failures = set()
        for row in payload["terminal_failures"]:
            fields(row, ("category", "task_id", "failure_kind"))
            if (
                not isinstance(row["category"], str)
                or not isinstance(row["task_id"], str)
                or row["failure_kind"] != "executor_response_error"
            ):
                raise ValueError("invalid terminal interface failure")
            identity = row["category"], row["task_id"]
            if identity in terminal_failures:
                raise ValueError("duplicate terminal failure identity")
            terminal_failures.add(identity)
        result = score_batch(
            payload["config"], predictions, terminal_failures=terminal_failures
        )
    json.dump(result, sys.stdout, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
