"""Explicit currency-service migration for the pinned ToolSandbox tool."""

import math

from tool_sandbox.tools import rapid_api_search_tools

OLD_CURRENCY_URL = "https://currency-converter18.p.rapidapi.com/api/v1/convert"
CURRENCY_HOST = "currency-conversion-and-exchange-rates.p.rapidapi.com"
CURRENCY_URL = f"https://{CURRENCY_HOST}/convert"
_official_request = rapid_api_search_tools.rapid_api_get_request


def currency_request(url, params, headers):
    if url != OLD_CURRENCY_URL:
        return _official_request(url=url, params=params, headers=headers)
    response = _official_request(
        url=CURRENCY_URL,
        params=params,
        headers={**headers, "X-RapidAPI-Host": CURRENCY_HOST},
    )
    if response["success"] is not True:
        raise RuntimeError("Currency conversion service reported failure")
    value = response["result"]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Currency conversion result must be a finite number")
    # Keep the official tool object, validation, schema and tool-trace wrapper.
    return {"result": {"convertedAmount": value}}


def install():
    """Repeated configure calls install the same hook without wrapping it again."""
    rapid_api_search_tools.rapid_api_get_request = currency_request
