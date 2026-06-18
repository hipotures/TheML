from __future__ import annotations

import json

from .client import AiRequest, AiResponse


class MockAiClient:
    def call(self, request: AiRequest) -> AiResponse:
        if request.role == "hypothesis":
            payload = {
                "title": "Baseline robust tabular preprocessing",
                "summary": "Create a small leakage-safe first ROOT hypothesis for tabular data.",
                "feature_family": "baseline_preprocessing",
                "feature_strategy": "Add bounded, target-free preprocessing and preserve row order.",
                "materialization_hint": "Autogluon mode should expose preprocess(df); legacy mode may run a full script.",
                "expected_signal": "Useful as a smoke-tested baseline before richer ROOT hypotheses.",
                "risk": "This mock hypothesis is intentionally generic and should be replaced by a real model backend.",
                "sources": [],
            }
        elif request.role == "code":
            payload = {"code": legacy_code() if "legacy" in request.prompt else autogluon_code()}
        else:
            payload = {"summary": "Mock response", "ok": True}
        return AiResponse(text=json.dumps(payload, indent=2), metadata={"backend": "mock"})


def autogluon_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import pandas as pd\n\n\n"
        "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = df.copy()\n"
        "    return out\n"
    )


def legacy_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "result = {'status': 'ok', 'metric': None, 'maximize': True}\n"
        "print('TML_RESULT_JSON:' + json.dumps(result, sort_keys=True))\n"
    )
