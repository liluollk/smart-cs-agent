"""
API Schema — 增强输入校验，防止注入和恶意输入。
使用 Pydantic field_validator 替代手动 validate_message。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


PROMPT_INJECTION_PATTERNS = [
    r"<\|.*\|>",
    r"<im_start>",
    r"<im_end>",
]

SQL_INJECTION_PATTERNS = [
    r"(?i)(\bSELECT\b.*\bFROM\b)",
    r"(?i)(\bDROP\b\s+\bTABLE\b)",
    r"(?i)(\bINSERT\b\s+\bINTO\b)",
    r"(?i)(\bDELETE\b\s+\bFROM\b)",
    r"(?i)(\bUPDATE\b\s+\w+\s+\bSET\b)",
    r"(?i)(--\s*$)",
    r"(?i)(\bUNION\b\s+\bSELECT\b)",
    r"(?i)(\bOR\b\s+1\s*=\s*1)",
    r"(?i)(\bAND\b\s+1\s*=\s*1)",
    r"(?i)(\bEXEC\b\s*\()",
    r"(?i)(\bEXECUTE\b\s*\()",
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="用户消息")
    user_id: str = Field(default="anonymous", max_length=64, pattern=r"^[a-zA-Z0-9_\-@.]+$")
    session_id: str = Field(default="", max_length=64, pattern=r"^[a-zA-Z0-9\-]*$")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("消息不能为空")

        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, stripped):
                raise ValueError("消息包含不安全的SQL模式")

        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, stripped, re.IGNORECASE):
                raise ValueError("消息包含疑似注入模式")

        if stripped.count("<") > 20 or stripped.count(">") > 20:
            raise ValueError("消息包含过多HTML标签")

        return stripped


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    intent: str
    response: str
    compliance_passed: bool