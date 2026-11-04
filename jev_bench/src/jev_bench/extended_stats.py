"""Formal official metrics and paired inference at each benchmark's sampling unit."""

from collections import Counter, defaultdict

import numpy as np

from .config import BENCHMARKS
from .cost import measured
from .runner import key, resolved, validate_attempts
from .stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, aggregate_cost, exact_permutation

PERMUTATION_DRAWS = 100000
EXACT_CLUSTER_LIMIT = 20


def cluster(item, benchmark):
    if benchmark == "apibank":
        source = item["source_id"]
        level = item["category"].split("_")[1]
        return (level, source["sample_id"] if level == "3" else source["file"])
    if benchmark == "toolsandbox":
        return item["family_id"]
    return item["task_id"]


def score(row, benchmark):
    if row is None:
        return None
    if benchmark == "toolhop":
        # Keep the existing named PA interface-failure rule; other missing scores
        # remain unresolved and are never filled with zero.
        return None if row["success"] is None else int(row["success"])
    if not row["official_scored"]:
        return None
    if benchmark == "apibank" and row["category"].endswith("_api"):
        return int(row["success"])
    return row["official_score"]["value"]


def metric_name(benchmark, category):
    if benchmark == "apibank":
        return "api_accuracy" if category.endswith("_api") else "rouge_l_f"
    return {
        "toolhop": "answer_correct",
        "toolsandbox": "similarity",
        "stabletoolbench": "solvable_pass_rate",
    }[benchmark]


def cluster_bootstrap(sums, sizes):
    """Resample whole clusters, retaining point weights and zero-difference clusters."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums, sizes = np.asarray(sums, dtype=float), np.asarray(sizes, dtype=int)
    estimates = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 1000):
        stop = min(start + 1000, BOOTSTRAP_DRAWS)
        indices = rng.integers(len(sums), size=(stop - start, len(sums)))
        estimates[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def cluster_permutation(sums):
    values = np.asarray([value for value in sums if value != 0], dtype=float)
    observed = abs(values.sum())
    # Include mathematical ties despite floating-point addition order.
    tolerance = 1e-12 * max(1, np.abs(values).sum())
    exact = len(values) <= EXACT_CLUSTER_LIMIT
    draws = 2 ** len(values) if exact else PERMUTATION_DRAWS
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    extreme = 0
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        if exact:
            masks = np.arange(start, stop, dtype=np.uint64)
            bits = (masks[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1
        else:
            bits = rng.integers(2, size=(stop - start, len(values)))
        simulated = (2 * bits.astype(float) - 1) @ values
        extreme += int(np.count_nonzero(np.abs(simulated) >= observed - tolerance))
    return {
        "p_value": extreme / draws if exact else (extreme + 1) / (draws + 1),
        "p_method": "exact_cluster_permutation"
        if exact
        else "monte_carlo_cluster_permutation",
        "permutation_draws": draws,
        "nonzero_clusters": len(values),
    }


def paired(latest, plan, treatment, reference, benchmark, category):
    left = {
        item["task_id"]: item
        for item in plan
        if item["condition"] == treatment and item["category"] == category
    }
    right = {
        item["task_id"]: item
        for item in plan
        if item["condition"] == reference and item["category"] == category
    }
    if left.keys() != right.keys():
        raise ValueError("formal comparisons require identical planned point coverage")
    result = {
        "treatment": treatment,
        "reference": reference,
        "category": category,
        "metric": metric_name(benchmark, category),
        "points": len(left),
        "complete": False,
        "difference": None,
        "p_value": None,
        "bootstrap_95_percentile": None,
    }
    sums, sizes = defaultdict(float), Counter()
    missing = 0
    for identity in sorted(left):
        item, other = left[identity], right[identity]
        group = cluster(item, benchmark)
        if group != cluster(other, benchmark):
            raise ValueError("canonical cluster differs across conditions")
        sizes[group] += 1
        a, b = (
            score(latest.get(key(item)), benchmark),
            score(latest.get(key(other)), benchmark),
        )
        if a is None or b is None:
            missing += 1
        else:
            sums[group] += a - b
    result.update(clusters=len(sizes), missing_pairs=missing)
    if not left or missing:
        result["withheld_reason"] = "incomplete_official_scores_or_plan"
        return result
    groups = list(sizes)
    differences = [sums[group] for group in groups]
    counts = [sizes[group] for group in groups]
    result.update(
        complete=True,
        difference=sum(differences) / sum(counts),
        bootstrap_95_percentile=cluster_bootstrap(differences, counts),
        bootstrap_draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    if benchmark == "toolhop":
        result.update(
            p_value=exact_permutation([int(value) for value in differences]),
            p_method="exact_task_permutation",
        )
    else:
        result.update(cluster_permutation(differences))
    return result


def holm(comparisons):
    """Keep the predeclared family size even when some tests lack scores."""
    ordered = sorted(
        (item for item in comparisons if item["p_value"] is not None),
        key=lambda item: item["p_value"],
    )
    previous = 0.0
    for index, item in enumerate(ordered):
        previous = max(previous, min(1.0, (len(comparisons) - index) * item["p_value"]))
        item["p_holm"] = previous
    for item in comparisons:
        if item["p_value"] is None:
            item["p_holm"] = None


def summarize(rows, plan, comparisons, *, benchmark):
    if not plan or any(item["trial"] != 0 for item in plan):
        raise ValueError("formal inference requires a nonempty one-trial plan")
    latest = validate_attempts(rows, plan)
    categories = BENCHMARKS[benchmark][2]
    conditions = {}
    for condition in dict.fromkeys(item["condition"] for item in plan):
        planned = [item for item in plan if item["condition"] == condition]
        selected = [row for row in latest.values() if row["condition"] == condition]
        scores, coverage = {}, {}
        for category in categories:
            items = [item for item in planned if item["category"] == category]
            values = [score(latest.get(key(item)), benchmark) for item in items]
            scored = sum(
                latest[key(item)]["official_scored"]
                for item in items
                if key(item) in latest
            )
            known = [value for value in values if value is not None]
            complete = bool(items) and len(known) == len(items)
            coverage[category] = {
                "planned": len(items),
                "official_scored": scored,
                "missing_scores": len(items) - scored,
                "counted": len(known),
                "complete": complete,
            }
            scores[category] = {
                "metric": metric_name(benchmark, category),
                "count": len(known),
                "mean": sum(known) / len(items) if complete else None,
            }
        summary = {
            "coverage_by_category": coverage,
            "scores": scores,
            "official_scored": sum(row["official_scored"] for row in selected),
            "failures_all_attempts": dict(
                Counter(
                    row["failure_kind"]
                    for row in rows
                    if row["condition"] == condition and row["failure_kind"]
                )
            ),
            "cost_all_attempts": aggregate_cost(
                [row for row in rows if row["condition"] == condition]
            ),
        }
        if benchmark == "toolhop":
            summary["tool_execution_timeout"] = measured(
                [
                    None
                    if row["failure_kind"] == "interrupted"
                    and "tool_error_counts" not in row
                    else row["tool_error_counts"]["tool_execution_timeout"]
                    for row in rows
                    if row["condition"] == condition
                ]
            )
        if benchmark == "apibank":
            from .benchmarks.apibank_scoring import sample_accuracy

            points = []
            for item in planned:
                if item["category"] == "level_3_api":
                    point = {"source_id": item["source_id"]}
                    row = latest.get(key(item))
                    if row is not None and row["official_scored"]:
                        point["correct"] = row["success"]
                    points.append(point)
            summary["level_3_sample_accuracy"] = sample_accuracy(points)
        if benchmark == "toolsandbox":
            summary["official_categories"] = None
            if coverage["scenarios"]["complete"]:
                from .benchmarks.toolsandbox_scores import describe

                summary["official_categories"] = describe(selected, planned, [])[
                    "conditions"
                ][condition]
        conditions[condition] = summary
    paired_results = [
        paired(latest, plan, *pair, benchmark, category)
        for pair in comparisons
        for category in categories
    ]
    holm(paired_results)
    return {
        "purpose": "formal_evaluation",
        "planned": len(plan),
        "attempts": len(rows),
        "resolved": sum(resolved(row) for row in latest.values()),
        "execution_coverage_complete": len(latest) == len(plan)
        and all(resolved(row) for row in latest.values()),
        "coverage_complete": all(
            value["complete"]
            for summary in conditions.values()
            for value in summary["coverage_by_category"].values()
        ),
        "conditions": conditions,
        "comparisons": paired_results,
        "multiplicity": {
            "method": "Holm",
            "family": "all declared primary pairs and metrics within this benchmark",
            "family_size": len(paired_results),
            "missing_tests_retain_family_size": True,
        },
        "definitions": {
            "difference": "treatment minus reference; official point weights preserved",
            "sampling_unit": {
                "toolhop": "task",
                "apibank": "source conversation within each of six categories",
                "toolsandbox": "base scenario family with all eight variants",
                "stabletoolbench": "query within each of six groups",
            }[benchmark],
            "bootstrap": f"pointwise 95% percentile; {BOOTSTRAP_DRAWS} cluster resamples; seed={BOOTSTRAP_SEED}; zero differences retained",
            "permutation": "two-sided; ToolHop exact; other benchmarks enumerate <=20 nonzero clusters, otherwise 100000 swaps with plus-one correction",
            "missing": "no zero fill or complete-case subset; withhold affected category means and comparisons; retain existing ToolHop named interface-failure counting",
            "secondary": "ToolSandbox official overlapping categories and API-Bank L3 Sample Accuracy are descriptive only; Stable judge repetitions are not separate samples",
            "cost": "all persisted attempts; unknown usage remains missing",
        },
    }
