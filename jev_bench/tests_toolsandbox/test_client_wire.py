"""Old ToolSandbox SDK sends the approved advisor body through a mock transport."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from jev_bench.advice import DeepSeekClient, advise
from jev_bench.config import PROJECT_ROOT, load_config
from openai import OpenAI


class ClientWireTests(unittest.TestCase):
    def test_nonstandard_thinking_fields_are_sent_by_the_installed_sdk(self):
        config = load_config(PROJECT_ROOT / "configs/toolsandbox_main_smoke.json")
        settings = config["advisors"]["deepseek"]
        observed = []

        def handler(request):
            observed.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "cpu",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Done"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        sdk = OpenAI(
            api_key="cpu-only",
            base_url="https://fixture.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        with (
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "cpu-only"}),
            patch("jev_bench.client.OpenAI", return_value=sdk),
        ):
            client = DeepSeekClient(settings, lambda *a: None)
            try:
                result = advise(
                    {"advice": "prefer"},
                    client,
                    [{"role": "user", "content": "fixture"}],
                    [],
                    [],
                )
            finally:
                client.close()
        self.assertEqual(result["prediction"]["top1"], "RESPOND")
        self.assertEqual(observed[0]["thinking"], {"type": "enabled"})
        self.assertEqual(observed[0]["reasoning_effort"], "high")
        self.assertNotIn("extra_body", observed[0])
