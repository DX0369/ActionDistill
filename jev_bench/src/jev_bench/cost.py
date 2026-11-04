"""Observed inference costs, including failed requests and simulated users."""

# Advisor cost fields by configured type. A field that does not apply to an
# advisor is omitted rather than reported as missing.
ADVISOR_FIELDS = {
    None: (),
    "jev_local": (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "model_latency_seconds",
        "truncated",
        "messages_dropped",
        "local_prompt_tokens",
    ),
    "typesafe": (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "model_latency_seconds",
        "truncated",
        "messages_dropped",
        "local_prompt_tokens",
        "render_latency_seconds",
        "remote_input_tokens",
        "remote_output_tokens",
    ),
    "deepseek": (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "model_latency_seconds",
        "remote_input_tokens",
        "remote_output_tokens",
        "reasoning_tokens",
    ),
}
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
# Fields measured only on requests that actually reached the remote provider.
PROVIDER_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model_latency_seconds",
    "remote_input_tokens",
    "remote_output_tokens",
    "reasoning_tokens",
)


def measured(values):
    known = [value for value in values if value is not None]
    return {
        "observed_total": sum(known),
        "known": len(known),
        "missing": len(values) - len(known),
    }


def local_advisor_cost(advisor):
    # Optional validated metadata is retained separately on malformed actions;
    # it is accounting evidence, never a substitute action prediction.
    predictions = [
        row["prediction"]
        if row["prediction"] is not None
        else row.get("validated_metadata")
        for row in advisor
    ]
    values = {
        "completion_tokens": [0 if p is not None else None for p in predictions],
        "latency_seconds": [row["latency_seconds"] for row in advisor],
        "model_latency_seconds": [
            p["latency_seconds"] if p is not None else None for p in predictions
        ],
        "truncated": [
            int(p["truncated"]) if p is not None else None for p in predictions
        ],
        "messages_dropped": [
            p["messages_dropped"] if p is not None else None for p in predictions
        ],
    }
    for field in ("prompt_tokens", "total_tokens", "local_prompt_tokens"):
        values[field] = [
            p["n_prompt_tokens"] if p is not None else None for p in predictions
        ]
    result = {
        "requests": len(advisor),
        "advisor_format_error_neutral": sum(
            row["fallback"] == "advisor_format_error_neutral" for row in advisor
        ),
        "over_length_after_drop_neutral": sum(
            row["fallback"] == "over_length_after_drop_neutral" for row in advisor
        ),
    }
    result.update(
        {field: measured(values[field]) for field in ADVISOR_FIELDS["jev_local"]}
    )
    return result


def remote_advisor_cost(advisor_type, advisor_requests):
    if any(row["advisor_type"] != advisor_type for row in advisor_requests):
        raise ValueError(
            "advisor request ledger differs from the configured advisor type"
        )
    provider = [row for row in advisor_requests if row["provider_called"]]
    result = {
        "requests": len(advisor_requests),
        "provider_requests": len(provider),
        "advisor_format_error_neutral": sum(
            row["fallback"] == "advisor_format_error_neutral"
            for row in advisor_requests
        ),
    }
    if advisor_type == "typesafe":
        result["over_length_after_drop_neutral"] = sum(
            row["fallback"] == "over_length_after_drop_neutral"
            for row in advisor_requests
        )
    if advisor_type == "deepseek":
        result["stable_thought_neutral"] = sum(
            row["fallback"] == "stable_thought_neutral" for row in advisor_requests
        )
    for field in ADVISOR_FIELDS[advisor_type]:
        rows = provider if field in PROVIDER_FIELDS else advisor_requests
        if field in USAGE_FIELDS:
            values = [
                row["usage"][field] if row["usage"] is not None else None
                for row in rows
            ]
        elif field == "truncated":
            values = [
                int(row[field]) if row[field] is not None else None for row in rows
            ]
        else:
            values = [row[field] for row in rows]
        result[field] = measured(values)
    return result


def costs(client, handler, user_client=None, *, advisor_type, advisor_requests=()):
    records = client.records if client is not None else []
    advisor = handler.advisor_records if handler is not None else []
    if advisor_type is None:
        if advisor or advisor_requests:
            raise ValueError("a condition without an advisor recorded advisor requests")
        advisor_cost = {"requests": 0}
    elif advisor_type == "jev_local":
        if advisor_requests:
            raise ValueError("local Jev has no remote request ledger")
        advisor_cost = local_advisor_cost(advisor)
    else:
        # Remote transport records are one-per-predict, independent of whether
        # the later executor response (and adoption record) reached disk.
        advisor_cost = remote_advisor_cost(advisor_type, advisor_requests)
    result = {
        "accounting_complete": True,
        "executor": {"requests": len(records)},
        "advisor": advisor_cost,
    }
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        result["executor"][field] = measured(
            [
                row["usage"][field] if row["usage"] is not None else None
                for row in records
            ]
        )
    result["executor"]["latency_seconds"] = measured(
        [row["latency_seconds"] for row in records]
    )
    result["user_simulator"] = completion_cost(user_client)
    return result


def completion_cost(client):
    records = client.records if client is not None else []
    result = {"requests": len(records)}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        result[field] = measured(
            [
                row["usage"][field] if row["usage"] is not None else None
                for row in records
            ]
        )
    result["latency_seconds"] = measured([row["latency_seconds"] for row in records])
    return result
