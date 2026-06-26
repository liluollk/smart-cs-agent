from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return val if val is not None else default


@dataclass
class Settings:
    env: str = field(default_factory=lambda: _env("APP_ENV", "development"))

    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen-plus"))
    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY", ""))
    dashscope_base_url: str = field(
        default_factory=lambda: _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))

    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))

    pg_host: str = field(default_factory=lambda: _env("PG_HOST") or _env("PGVECTOR_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: int(_env("PG_PORT") or _env("PGVECTOR_PORT", "5432")))
    pg_database: str = field(default_factory=lambda: _env("PG_DATABASE") or _env("PGVECTOR_DB", "smartcs"))
    pg_user: str = field(default_factory=lambda: _env("PG_USER") or _env("PGVECTOR_USER", "postgres"))
    pg_password: str = field(default_factory=lambda: _env("PG_PASSWORD") or _env("PGVECTOR_PASSWORD", ""))

    rate_limit_per_minute: int = field(default_factory=lambda: int(_env("RATE_LIMIT_PER_MINUTE", "60")))
    tool_timeout_seconds: float = field(default_factory=lambda: float(_env("TOOL_TIMEOUT_SECONDS", "30")))
    embedding_cache_size: int = field(default_factory=lambda: int(_env("EMBEDDING_CACHE_SIZE", "256")))

    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    def validate(self) -> list[str]:
        warnings = []
        if not self.dashscope_api_key:
            warnings.append("DASHSCOPE_API_KEY 未设置，LLM 调用将失败")
        if not self.redis_url:
            warnings.append("REDIS_URL 未设置，短期记忆将使用内存模式")
        if not self.pg_password:
            warnings.append("PG_PASSWORD 未设置，PostgreSQL 不可用，服务将无法启动")
        if self.is_production:
            if self.llm_temperature > 0.3:
                warnings.append(f"生产环境 LLM_TEMPERATURE={self.llm_temperature} 过高，建议 ≤ 0.3")
            if self.rate_limit_per_minute > 120:
                warnings.append(f"生产环境 RATE_LIMIT_PER_MINUTE={self.rate_limit_per_minute} 过高，建议 ≤ 120")
        return warnings


settings = Settings()