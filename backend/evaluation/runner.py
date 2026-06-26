"""
A/B 评估体系 — 批量运行测试用例，统计准确率
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx

from evaluation.test_cases import TEST_CASES


@dataclass
class EvaluationResult:
    test_id: str
    category: str
    passed: bool
    intent_match: bool
    keyword_matches: dict[str, bool]
    compliance_match: bool
    response: str
    actual_intent: str
    actual_compliance: bool
    duration_ms: float
    errors: list[str] = field(default_factory=list)


class EvaluationRunner:

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: list[EvaluationResult] = []

    async def run_all(self):
        async with httpx.AsyncClient(timeout=120.0) as client:
            for tc in TEST_CASES:
                result = await self._run_case(client, tc)
                self.results.append(result)
                status = "✅" if result.passed else "❌"
                print(f"{status} {result.test_id} [{result.category}] {result.duration_ms:.0f}ms")
                if result.errors:
                    for e in result.errors:
                        print(f"   ⚠ {e}")

    async def _run_case(self, client: httpx.AsyncClient, tc: dict[str, Any]) -> EvaluationResult:
        start = time.time()
        errors = []

        try:
            resp = await client.post(f"{self.base_url}/v1/chat", json={
                "message": tc["message"],
                "user_id": f"eval_{tc['id']}",
            })

            if resp.status_code != 200:
                ""
                try:
                    detail = resp.json().get("detail", "") or str(resp.text)
                except Exception:
                    detail = str(resp.text)
                duration = (time.time() - start) * 1000
                compliance_match = tc["expected_compliance"] is False
                passed = compliance_match
                return EvaluationResult(
                    test_id=tc["id"], category=tc["category"], passed=passed,
                    intent_match=False, keyword_matches={}, compliance_match=compliance_match,
                    response=detail[:200], actual_intent="error", actual_compliance=False,
                    duration_ms=duration, errors=[f"HTTP {resp.status_code}: {detail[:100]}"],
                )

            data = resp.json()
        except Exception as e:
            duration = (time.time() - start) * 1000
            return EvaluationResult(
                test_id=tc["id"], category=tc["category"], passed=False,
                intent_match=False, keyword_matches={}, compliance_match=False,
                response=str(e), actual_intent="error", actual_compliance=False,
                duration_ms=duration, errors=[f"API error: {e}"],
            )

        duration = (time.time() - start) * 1000
        response = data.get("response", "")
        actual_intent = data.get("intent", "")
        actual_compliance = data.get("compliance_passed", True)

        intent_match = actual_intent == tc["expected_intent"]
        if not intent_match:
            errors.append(f"意图不匹配: 期望 {tc['expected_intent']}, 实际 {actual_intent}")

        keyword_matches = {}
        for kw in tc["expected_keywords"]:
            keyword_matches[kw] = kw in response
            if not keyword_matches[kw]:
                errors.append(f"关键词缺失: '{kw}'")

        compliance_match = actual_compliance == tc["expected_compliance"]
        if not compliance_match:
            errors.append(f"合规不匹配: 期望 {tc['expected_compliance']}, 实际 {actual_compliance}")

        all_keywords_ok = all(keyword_matches.values())

        if tc["expected_compliance"] is False:
            passed = compliance_match
        else:
            passed = intent_match and all_keywords_ok and compliance_match

        return EvaluationResult(
            test_id=tc["id"], category=tc["category"], passed=passed,
            intent_match=intent_match, keyword_matches=keyword_matches,
            compliance_match=compliance_match, response=response[:200],
            actual_intent=actual_intent, actual_compliance=actual_compliance,
            duration_ms=duration, errors=errors,
        )

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        intent_accuracy = sum(1 for r in self.results if r.intent_match) / max(total, 1)
        compliance_accuracy = sum(1 for r in self.results if r.compliance_match) / max(total, 1)
        avg_duration = sum(r.duration_ms for r in self.results) / max(total, 1)

        by_category = {}
        for r in self.results:
            prefix = r.category.split("-")[0] if "-" in r.category else r.category
            bucket = by_category.setdefault(prefix, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if r.passed:
                bucket["passed"] += 1

        return {
            "total": total,
            "passed": passed,
            "accuracy": f"{passed / max(total, 1) * 100:.1f}%",
            "intent_accuracy": f"{intent_accuracy * 100:.1f}%",
            "compliance_accuracy": f"{compliance_accuracy * 100:.1f}%",
            "avg_duration_ms": f"{avg_duration:.0f}",
            "by_category": {k: f"{v['passed']}/{v['total']}" for k, v in by_category.items()},
        }


async def main():
    runner = EvaluationRunner()
    await runner.run_all()
    print("\n" + "=" * 50)
    summary = runner.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())