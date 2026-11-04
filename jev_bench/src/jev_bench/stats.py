"""Official category means, task-paired tests and stratified confidence intervals."""

import math
from collections import Counter
from functools import reduce

import numpy as np

from .config import BENCHMARKS
from .cost import measured
from .runner import key, resolved, validate_attempts

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261015


def exact_permutation(differences):
    if any(type(value) is not int for value in differences):
        raise ValueError("exact permutation requires integer task differences")
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    divisor = reduce(math.gcd, map(abs, nonzero))
    values = [value // divisor for value in nonzero]
    observed = abs(sum(values))
    distribution = Counter({0: 1})
    for value in values:
        updated = Counter()
        for total, count in distribution.items():
            updated[total + value] += count
            updated[total - value] += count
        distribution = updated
    return sum(
        count for total, count in distribution.items() if abs(total) >= observed
    ) / (2 ** len(values))


def task_differences(latest, plan, treatment, reference, categories):
    differences = {}
    counts = {}
    for category in categories:
        task_ids = sorted(
            {item["task_id"] for item in plan if item["category"] == category}
        )
        if not task_ids:
            raise ValueError("all configured benchmark categories must be present")
        values = []
        for task_id in task_ids:
            left = [
                latest[key(item)]
                for item in plan
                if item["condition"] == treatment
                and item["task_id"] == task_id
                and item["category"] == category
            ]
            right = [
                latest[key(item)]
                for item in plan
                if item["condition"] == reference
                and item["task_id"] == task_id
                and item["category"] == category
            ]
            if not left or {r["trial"] for r in left} != {r["trial"] for r in right}:
                raise ValueError(
                    "paired comparisons require identical task/trial coverage"
                )
            if any(type(row["success"]) is not bool for row in left + right):
                raise ValueError("unresolved episodes cannot enter paired statistics")
            counts[(category, task_id)] = len(left)
            values.append(
                sum(row["success"] for row in left)
                - sum(row["success"] for row in right)
            )
        differences[category] = values
    if len(set(counts.values())) != 1:
        raise ValueError("all tasks must have the same configured number of trials")
    return differences, next(iter(counts.values()))


def stratified_bootstrap(differences, trials, category_weights):
    """Resample paired tasks within each category, retaining all their trials."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.zeros(BOOTSTRAP_DRAWS)
    for category, weight in category_weights.items():
        values = np.asarray(differences[category], dtype=float) / trials
        indices = rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        estimates += values[indices].mean(axis=1) * weight
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def paired(latest, plan, treatment, reference, *, benchmark):
    if benchmark not in ("bfcl", "tau2"):
        raise ValueError("boolean paired tests are only defined for BFCL/tau2")
    categories = BENCHMARKS[benchmark][2]
    differences, trials = task_differences(
        latest, plan, treatment, reference, categories
    )
    sizes = [len(values) for values in differences.values()]
    multiple = math.lcm(*sizes)
    weights = [
        value * (multiple // len(values) if benchmark == "bfcl" else 1)
        for values in differences.values()
        for value in values
    ]
    all_values = [value for values in differences.values() for value in values]
    category_weights = {
        category: 1 / len(categories)
        if benchmark == "bfcl"
        else len(values) / len(all_values)
        for category, values in differences.items()
    }
    return {
        "treatment": treatment,
        "reference": reference,
        "tasks": len(all_values),
        "trials": trials,
        "difference": sum(
            sum(values) / len(values) / trials * category_weights[category]
            for category, values in differences.items()
        ),
        "p_exact_task_permutation": exact_permutation(weights),
        "bootstrap_95_percentile": stratified_bootstrap(
            differences, trials, category_weights
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "improved_tasks": sum(value > 0 for value in all_values),
        "regressed_tasks": sum(value < 0 for value in all_values),
        "unchanged_tasks": sum(value == 0 for value in all_values),
    }


def sum_measured(rows, participant, field):
    return {
        key: sum(row["cost"][participant][field][key] for row in rows)
        for key in ("observed_total", "known", "missing")
    }


def aggregate_cost(rows):
    result = {}
    participants = ["executor", "user_simulator"]
    for participant in ("tool_simulator", "judge"):
        if any(participant in row["cost"] for row in rows):
            if not all(participant in row["cost"] for row in rows):
                raise ValueError(f"missing {participant} ledger in some attempts")
            participants.append(participant)
    for participant in participants:
        result[participant] = {
            "requests": sum(row["cost"][participant]["requests"] for row in rows)
        }
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_seconds",
        ):
            result[participant][field] = sum_measured(rows, participant, field)
    # Rows aggregate within one condition, so they share one advisor type and field set.
    if len({frozenset(row["cost"]["advisor"]) for row in rows}) > 1:
        raise ValueError("aggregated rows report different advisor cost fields")
    fields = list(rows[0]["cost"]["advisor"]) if rows else ["requests"]
    advisor = result["advisor"] = {}
    for field in fields:
        if field in (
            "requests",
            "provider_requests",
            "over_length_after_drop_neutral",
            "advisor_format_error_neutral",
            "stable_thought_neutral",
        ):
            advisor[field] = sum(row["cost"]["advisor"][field] for row in rows)
        else:
            advisor[field] = sum_measured(rows, "advisor", field)
    if "truncated" in advisor:
        truncation = advisor["truncated"]
        advisor["truncated_request_rate"] = (
            truncation["observed_total"] / truncation["known"]
            if truncation["known"]
            else None
        )
        advisor["attempts_with_truncation"] = sum(
            row["cost"]["advisor"]["truncated"]["observed_total"] > 0 for row in rows
        )
    result["attempts_with_incomplete_accounting"] = sum(
        not row["cost"]["accounting_complete"] for row in rows
    )
    result["executor"]["output_limit_steps"] = sum(
        "output_limit" in event["flags"]
        for row in rows
        for event in row["response_anomalies"]
    )
    result["executor"]["empty_response_steps"] = sum(
        "empty_response" in event["flags"]
        for row in rows
        for event in row["response_anomalies"]
    )
    for field in ("duration_seconds", "decision_steps", "tool_call_count"):
        known = [row[field] for row in rows if row[field] is not None]
        result[field] = {
            "observed_total": sum(known),
            "known": len(known),
            "missing": len(rows) - len(known),
        }
    return result


def pass_power(rows):
    from tau2.metrics.agent_metrics import pass_hat_k

    grouped = {}
    for row in rows:
        grouped.setdefault((row["category"], row["task_id"]), []).append(row["success"])
    trial_counts = {len(values) for values in grouped.values()}
    if len(trial_counts) != 1:
        raise ValueError("pass^k requires the same complete trial count for every task")
    trials = trial_counts.pop()
    return {
        str(k): sum(pass_hat_k(trials, sum(values), k) for values in grouped.values())
        / len(grouped)
        for k in range(1, trials + 1)
    }


def summarize(rows, plan, comparisons, *, benchmark, formal=False):
    if formal:
        from .extended_stats import summarize as summarize_formal

        return summarize_formal(rows, plan, comparisons, benchmark=benchmark)
    if benchmark in ("apibank", "toolsandbox", "stabletoolbench"):
        if comparisons:
            raise ValueError("only descriptive engineering summaries are approved")
        latest = validate_attempts(rows, plan)
        conditions = {}
        for condition in dict.fromkeys(item["condition"] for item in plan):
            selected = [row for row in latest.values() if row["condition"] == condition]
            planned_categories = Counter(
                item["category"] for item in plan if item["condition"] == condition
            )
            scored_categories = Counter(
                row["category"] for row in selected if row["official_scored"]
            )
            failures = Counter(
                row["category"]
                for row in selected
                if row["failure_kind"] == "executor_response_error"
            )
            metrics = {}
            for row in selected:
                if not row["official_scored"]:
                    continue
                if benchmark == "apibank" and row["category"] in {
                    f"level_{level}_api" for level in (1, 2, 3)
                }:
                    score = {"metric": "api_accuracy", "value": int(row["success"])}
                else:
                    score = row["official_score"]
                metrics.setdefault(row["category"], []).append(score)
            conditions[condition] = {
                "official_scored": sum(row["official_scored"] for row in selected),
                "protocol_failures": sum(failures.values()),
                "coverage_by_category": {
                    category: {
                        "planned": count,
                        "official_scored": scored_categories[category],
                        "missing_scores": count - scored_categories[category],
                        "protocol_failures": failures[category],
                    }
                    for category, count in planned_categories.items()
                },
                "scores": {
                    category: {
                        "metric": values[0]["metric"],
                        "count": len(values),
                        "mean": sum(value["value"] for value in values) / len(values)
                        if not failures[category]
                        else None,
                    }
                    for category, values in metrics.items()
                },
                "cost_all_attempts": aggregate_cost(
                    [row for row in rows if row["condition"] == condition]
                ),
            }
        return {
            "purpose": "engineering_smoke_only",
            "planned": len(plan),
            "attempts": len(rows),
            "resolved": sum(resolved(row) for row in latest.values()),
            "execution_coverage_complete": bool(plan)
            and len(latest) == len(plan)
            and all(resolved(row) for row in latest.values()),
            "coverage_complete": bool(plan)
            and len(latest) == len(plan)
            and all(row["official_scored"] for row in latest.values()),
            "conditions": conditions,
            "comparisons": [],
        }
    if benchmark == "toolhop":
        if comparisons:
            raise ValueError("ToolHop inferential comparisons are not approved")
        latest = validate_attempts(rows, plan)
        return {
            "purpose": "engineering_smoke_only",
            "coverage_complete": len(latest) == len(plan)
            and all(row["success"] is not None for row in latest.values()),
            "planned": len(plan),
            "attempts": len(rows),
            "resolved": sum(row["success"] is not None for row in latest.values()),
            "conditions": {
                condition: {
                    "official_scored": sum(
                        row["official_scored"]
                        for row in latest.values()
                        if row["condition"] == condition
                    ),
                    "official_correct": sum(
                        row["success"] is True
                        for row in latest.values()
                        if row["condition"] == condition
                    ),
                    "cost_all_attempts": aggregate_cost(
                        [row for row in rows if row["condition"] == condition]
                    ),
                    "tool_execution_timeout": measured(
                        [
                            None
                            if row["failure_kind"] == "interrupted"
                            and "tool_error_counts" not in row
                            else row["tool_error_counts"]["tool_execution_timeout"]
                            for row in rows
                            if row["condition"] == condition
                        ]
                    ),
                }
                for condition in dict.fromkeys(item["condition"] for item in plan)
            },
        }
    if benchmark not in ("bfcl", "tau2"):
        raise ValueError("boolean summaries are only defined for BFCL/tau2")
    benchmark_categories = BENCHMARKS[benchmark][2]
    latest = validate_attempts(rows, plan)
    complete = len(latest) == len(plan) and all(
        row["success"] is not None for row in latest.values()
    )
    conditions = {}
    for condition in dict.fromkeys(item["condition"] for item in plan):
        selected = [row for row in latest.values() if row["condition"] == condition]
        attempts = [row for row in rows if row["condition"] == condition]
        categories = {}
        for category in benchmark_categories:
            planned = sum(
                item["condition"] == condition and item["category"] == category
                for item in plan
            )
            observed = [row for row in selected if row["category"] == category]
            scored = [row for row in observed if row["success"] is not None]
            successes = sum(row["success"] for row in scored)
            categories[category] = {
                "planned": planned,
                "resolved": len(scored),
                "successful": successes,
                "official_scored": sum(row["official_scored"] for row in observed),
                "accuracy": successes / planned
                if planned and len(scored) == planned
                else None,
                "cost_all_attempts": aggregate_cost(
                    [row for row in attempts if row["category"] == category]
                ),
            }
            if benchmark == "tau2":
                categories[category]["pass_power"] = (
                    pass_power(scored) if planned and len(scored) == planned else None
                )
        aggregate = None
        if complete:
            if benchmark == "bfcl":
                # A complete smoke plan can omit categories; the four-category
                # metric remains undefined until every category has a score.
                if all(x["accuracy"] is not None for x in categories.values()):
                    aggregate = sum(x["accuracy"] for x in categories.values()) / len(
                        benchmark_categories
                    )
            else:
                aggregate = sum(row["success"] for row in selected) / len(selected)
        conditions[condition] = {
            "categories": categories,
            (
                "official_multi_turn_accuracy"
                if benchmark == "bfcl"
                else "official_success_rate"
            ): aggregate,
            "failures_all_attempts": dict(
                Counter(row["failure_kind"] for row in attempts if row["failure_kind"])
            ),
            "cost_all_attempts": aggregate_cost(attempts),
        }
        if benchmark == "tau2":
            conditions[condition]["pass_power"] = (
                pass_power(selected) if complete else None
            )
    return {
        "coverage_complete": complete,
        "planned": len(plan),
        "attempts": len(rows),
        "resolved": sum(row["success"] is not None for row in latest.values()),
        "conditions": conditions,
        "paired_comparisons": [
            paired(latest, plan, *pair, benchmark=benchmark) for pair in comparisons
        ]
        if complete
        else [],
        "definitions": {
            "success": (
                "official score.valid"
                if benchmark == "bfcl"
                else "official is_successful(reward)"
            )
            + ", or a named protocol-counted executor failure",
            "aggregate": "equal mean of the four BFCL multi-turn category accuracies"
            if benchmark == "bfcl"
            else "equal task weights across selected base tasks; official pass^k formula, not pass@k",
            "cost": "all persisted attempts including failures; missing usage is never zero; interrupted request accounting may be incomplete",
            "confidence_interval": (
                "95% percentile interval for treatment minus reference; resample paired tasks "
                "within each category, retain all trials; "
                + (
                    "equal category weights; "
                    if benchmark == "bfcl"
                    else "fixed original domain task proportions; "
                )
                + f"{BOOTSTRAP_DRAWS} draws, seed={BOOTSTRAP_SEED}"
            ),
        },
    }
