"""Official continuous scores; descriptive CPU checks, no inferred success or p-values."""

from statistics import mean

from ..config import integer, number


def validate_score(score):
    for field in ("similarity", "milestone_similarity", "minefield_similarity"):
        number(score[field], maximum=1)
    integer(score["turn_count"])
    expected = score["milestone_similarity"] * int(score["minefield_similarity"] == 0)
    if score["similarity"] != expected:
        raise ValueError("similarity contradicts the official milestone/minefield rule")


def describe(rows, plan, comparisons):
    """Describe a fully scored fixture grid using official overlapping categories.

    Missing/failed cases are not assigned zero or removed from the denominator.
    This deliberately has no attempt recovery, bootstrap or significance test.
    """
    from tool_sandbox.cli.utils import get_category_summary
    from tool_sandbox.common.execution_context import ScenarioCategories

    identity = lambda row: (row["condition"], row["task_id"], row["trial"])
    expected = {identity(item) for item in plan}
    if not expected or len(expected) != len(plan):
        raise ValueError("fixture plan must be nonempty and unique")
    trials = {}
    for condition, scenario, trial in expected:
        integer(trial)
        trials.setdefault(condition, {}).setdefault(scenario, set()).add(trial)
    for scenarios in trials.values():
        if len({frozenset(values) for values in scenarios.values()}) != 1:
            raise ValueError(
                "fixture trial sets must match across scenarios; no reweighting"
            )
    indexed = {}
    metadata = {}
    official_categories = {str(category) for category in ScenarioCategories}
    for row in rows:
        integer(row["trial"])
        key = identity(row)
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicate fixture result")
        if row["official_scored"] is not True:
            raise ValueError("unscored fixture: formal failure handling is pending")
        validate_score(row)
        if not isinstance(row["categories"], list) or not row["categories"]:
            raise ValueError("official categories are required")
        categories = row["categories"]
        if (
            any(not isinstance(category, str) for category in categories)
            or len(set(categories)) != len(categories)
            or not set(categories) <= official_categories
        ):
            raise ValueError("categories must be distinct official labels")
        for field in ("family_id", "variant"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"canonical {field} is required")
        current = (categories, row["family_id"], row["variant"])
        previous = metadata.setdefault(row["task_id"], current)
        if previous != current:
            raise ValueError("scenario metadata differs across conditions or trials")
        indexed[key] = row
    if indexed.keys() != expected:
        raise ValueError("incomplete fixture coverage")
    conditions = list(dict.fromkeys(item["condition"] for item in plan))
    summaries = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            str(category): {
                "count": len(values["similarity"]),
                "similarity": mean(values["similarity"]),
                "turn_count": mean(values["turn_count"]),
            }
            for category, values in get_category_summary(selected).items()
        }
    paired = []
    for treatment, reference in comparisons:
        left = {key[1:]: row for key, row in indexed.items() if key[0] == treatment}
        right = {key[1:]: row for key, row in indexed.items() if key[0] == reference}
        if not left or left.keys() != right.keys() or treatment == reference:
            raise ValueError("comparison requires exactly matched scenario/trial pairs")
        paired.append(
            {
                "treatment": treatment,
                "reference": reference,
                "pairs": len(left),
                "mean_similarity_difference": mean(
                    left[key]["similarity"] - right[key]["similarity"] for key in left
                ),
            }
        )
    return {
        "conditions": summaries,
        "comparisons": paired,
        "inference": "not_configured",
    }
