"""
合规检查 Agent — 分为前置检查和后置检查两个阶段。

前置检查（输入合规）：在 fan_out 之前检查用户输入，
  若包含敏感信息则直接标记，阻断后续 Agent 处理。

后置检查（输出合规）：在 fan_out 之后检查 Agent 输出，
  检测违规用语和 PII 泄露，进行脱敏或拦截。

改进：
- 规则可配置化（通过构造函数覆盖默认规则）
- 违规分级：warn（警告）/ block（阻断）
- 完善日志记录
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from tracing.collector import get_collector

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRules:
    warn_terms: list[str] = field(default_factory=lambda: [
        "预期收益率", "最高收益",
    ])
    block_terms: list[str] = field(default_factory=lambda: [
        "保证收益", "稳赚不赔", "零风险", "保本保息",
        "承诺回报",
        "内部消息", "内幕", "暗箱操作",
    ])
    sensitive_input_keywords: list[str] = field(default_factory=lambda: [
        "自杀", "自残", "杀", "死",
        "黑客", "攻击", "DDoS", "ddos",
        "漏洞", "注入", "exploit",
    ])
    prompt_injection_patterns: list[str] = field(default_factory=lambda: [
        r"忽略.*指令",
        r"ignore.*instruction",
        r"你.*是.*一个.*无限制",
        r"假装.*你.*是",
        r"pretend.*you.*are",
        r"忘记.*规则",
        r"forget.*rule",
        r"不要.*限制",
        r"不要.*拒绝",
        r"do.*not.*restrict",
        r"你.*现在.*是",
        r"you.*are.*now",
        r"角色扮演",
        r"role.?play",
        r"越狱",
        r"jailbreak",
        r"DAN.*模式",
        r"开发者模式",
        r"developer.*mode",
    ])
    sql_injection_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b",
        r"(?i)(UNION\s+SELECT|--\s|\bOR\b.*=.*\bOR\b)",
        r"'.*'.*=",
        r"\bDROP\s+TABLE\b",
        r"\bALTER\s+TABLE\b",
        r"\bTRUNCATE\b",
    ])
    xss_patterns: list[str] = field(default_factory=lambda: [
        r"<script.*?>",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<iframe",
        r"<img[^>]+onerror",
        r"<\?xml",
    ])
    pii_patterns: dict[str, str] = field(default_factory=lambda: {
        "phone": r"1[3-9]\d{9}",
        "id_card": r"\d{17}[\dXx]",
        "bank_card": r"\d{16,19}",
    })
    max_message_length: int = 2000


class ComplianceAgent:

    def __init__(self, rules: ComplianceRules | None = None):
        self._rules = rules or ComplianceRules()

    def _check_terms(self, content: str) -> list[dict[str, str]]:
        violations = []
        for t in self._rules.warn_terms:
            if t in content:
                violations.append({"term": t, "severity": "warn", "message": f"警告用语: {t}"})
        for t in self._rules.block_terms:
            if t in content:
                violations.append({"term": t, "severity": "block", "message": f"违规用语: {t}"})
        return violations

    def _check_pii(self, content: str) -> list[dict[str, str]]:
        violations = []
        label_map = {"phone": "手机号", "id_card": "身份证号", "bank_card": "银行卡号"}
        for pii_type, pattern in self._rules.pii_patterns.items():
            if re.search(pattern, content):
                label = label_map.get(pii_type, pii_type)
                violations.append({"type": pii_type, "severity": "block", "message": f"PII泄露: {label}"})
        return violations

    def _check_sensitive_input(self, content: str) -> list[dict[str, str]]:
        violations = []
        content_lower = content.lower()
        for kw in self._rules.sensitive_input_keywords:
            if kw.lower() in content_lower:
                violations.append({"keyword": kw, "severity": "block", "message": f"敏感内容: {kw}"})
        return violations

    def _mask_pii(self, content: str) -> str:
        def mask(m):
            t = m.group()
            if len(t) <= 4:
                return "****"
            return t[:3] + "*" * (len(t) - 6) + t[-3:]

        result = content
        for pattern in self._rules.pii_patterns.values():
            result = re.sub(pattern, mask, result)
        return result

    async def pre_check(self, state: dict[str, Any]) -> dict[str, Any]:
        collector = get_collector()
        trace_id = state.get("trace_id", "")
        span = collector.start_span("compliance_pre", trace_id=trace_id)

        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""

        violations = []

        if not user_message or not user_message.strip():
            violations.append({"severity": "block", "message": "空消息"})

        violations.extend(self._check_sensitive_input(user_message))

        content_lower = user_message.lower()
        for pattern in self._rules.prompt_injection_patterns:
            if re.search(pattern, content_lower):
                violations.append({"severity": "block", "message": f"Prompt注入: {pattern}"})
                break

        for pattern in self._rules.sql_injection_patterns:
            if re.search(pattern, content_lower):
                violations.append({"severity": "block", "message": f"SQL注入: {pattern}"})
                break

        for pattern in self._rules.xss_patterns:
            if re.search(pattern, content_lower):
                violations.append({"severity": "block", "message": f"XSS: {pattern}"})
                break

        if len(user_message) > self._rules.max_message_length:
            violations.append({"severity": "block", "message": f"消息过长: {len(user_message)}字符"})

        has_block = any(v.get("severity") == "block" for v in violations)
        has_warn = any(v.get("severity") == "warn" for v in violations)

        if has_block:
            logger.warning("前置合规阻断: %s", [v["message"] for v in violations])
        elif has_warn:
            logger.info("前置合规警告: %s", [v["message"] for v in violations])

        collector.end_span(span)
        return {
            **state,
            "input_compliance_blocked": has_block,
            "input_compliance_violations": [v["message"] for v in violations],
        }

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        collector = get_collector()
        trace_id = state.get("trace_id", "")
        span = collector.start_span("compliance_post", trace_id=trace_id)

        sub_results = state.get("sub_results", {})
        content = "\n".join(v for k, v in sub_results.items() if isinstance(v, str))

        if not content.strip():
            collector.end_span(span)
            return {**state, "compliance_passed": True}

        term_violations = self._check_terms(content)
        pii_violations = self._check_pii(content)
        all_violations = term_violations + pii_violations

        has_block = any(v.get("severity") == "block" for v in all_violations)
        passed = len(all_violations) == 0

        if has_block:
            logger.warning("后置合规阻断: %s", [v["message"] for v in all_violations])
            for key in sub_results:
                if isinstance(sub_results[key], str):
                    sub_results[key] = self._mask_pii(sub_results[key])
        elif all_violations:
            logger.info("后置合规警告: %s", [v["message"] for v in all_violations])

        collector.end_span(span)
        return {
            **state,
            "compliance_passed": not has_block,
            "sub_results": {
                **sub_results,
                "compliance": {
                    "passed": passed,
                    "has_warnings": any(v.get("severity") == "warn" for v in all_violations),
                    "violations": [v["message"] for v in all_violations],
                },
            },
        }