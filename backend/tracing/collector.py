"""
全链路追踪 — Span 收集器
内存收集 + 文件持久化，服务重启后可恢复 Trace 数据。
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    trace_id: str
    agent_name: str
    method: str
    start_time: float
    end_time: float = 0.0
    status: str = "running"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000


@dataclass
class ToolCallRecord:
    tool_name: str
    trace_id: str
    agent_name: str
    session_id: str
    success: bool
    duration_ms: float
    error: str | None = None
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)


class SpanCollector:
    """内存 Span 收集器，保留最近 N 条 Trace，并持久化到文件"""

    def __init__(self, max_traces: int = 100, persist_dir: str | None = None,
                 max_tool_records: int = 5000):
        self._traces: dict[str, list[Span]] = defaultdict(list)
        self._max_traces = max_traces
        self._active_spans: dict[str, Span] = {}
        self._persist_dir = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._tool_records: list[ToolCallRecord] = []
        self._max_tool_records = max_tool_records

    def start_span(self, agent_name: str, method: str = "process",
                   trace_id: str | None = None, parent_id: str | None = None) -> Span:
        if trace_id is None:
            trace_id = uuid.uuid4().hex[:12]
        span = Span(
            span_id=uuid.uuid4().hex[:8],
            parent_id=parent_id,
            trace_id=trace_id,
            agent_name=agent_name,
            method=method,
            start_time=time.time(),
        )
        self._active_spans[span.span_id] = span
        return span

    def end_span(self, span: Span, error: str | None = None,
                 token_usage: dict[str, int] | None = None):
        span.end_time = time.time()
        span.status = "error" if error else "success"
        span.error = error
        if token_usage:
            span.prompt_tokens = token_usage.get("prompt_tokens", 0)
            span.completion_tokens = token_usage.get("completion_tokens", 0)
            span.total_tokens = token_usage.get("total_tokens", 0)
        self._active_spans.pop(span.span_id, None)
        self._traces[span.trace_id].append(span)

        if len(self._traces) > self._max_traces:
            oldest = min(self._traces.keys(), key=lambda k: self._traces[k][0].start_time if self._traces[k] else 0)
            del self._traces[oldest]

        self._persist_trace(span.trace_id)

    def add_token_usage(self, span_id: str, token_usage: dict[str, int]):
        span = self._active_spans.get(span_id)
        if span:
            span.prompt_tokens += token_usage.get("prompt_tokens", 0)
            span.completion_tokens += token_usage.get("completion_tokens", 0)
            span.total_tokens += token_usage.get("total_tokens", 0)

    def record_tool_call(self, tool_name: str, trace_id: str, agent_name: str,
                         session_id: str, success: bool, duration_ms: float,
                         error: str | None = None, retry_count: int = 0):
        record = ToolCallRecord(
            tool_name=tool_name,
            trace_id=trace_id,
            agent_name=agent_name,
            session_id=session_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            retry_count=retry_count,
        )
        self._tool_records.append(record)
        if len(self._tool_records) > self._max_tool_records:
            self._tool_records = self._tool_records[-self._max_tool_records:]

    def get_tool_metrics(self, tool_name: str | None = None,
                         time_window_seconds: float | None = None) -> dict:
        records = self._get_filtered_records(tool_name, time_window_seconds)

        if not records:
            return {"total_calls": 0, "by_tool": {}, "summary": "暂无工具调用记录"}

        by_tool: dict[str, dict] = {}
        for r in records:
            bucket = by_tool.setdefault(r.tool_name, {
                "total": 0, "success": 0, "failed": 0,
                "durations": [], "errors": [],
            })
            bucket["total"] += 1
            if r.success:
                bucket["success"] += 1
            else:
                bucket["failed"] += 1
                if r.error:
                    bucket["errors"].append(r.error)

            bucket["durations"].append(r.duration_ms)

        result = {
            "total_calls": len(records),
            "success_rate": round(
                sum(1 for r in records if r.success) / len(records) * 100, 1
            ),
            "by_tool": {},
        }

        for name, bucket in by_tool.items():
            durations = bucket["durations"]
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            result["by_tool"][name] = {
                "total": bucket["total"],
                "success": bucket["success"],
                "failed": bucket["failed"],
                "success_rate": round(bucket["success"] / bucket["total"] * 100, 1),
                "avg_duration_ms": round(sum(durations) / n, 1),
                "p50_duration_ms": round(sorted_durations[n // 2], 1),
                "p99_duration_ms": (
                    round(sorted_durations[int(n * 0.99)], 1)
                    if n >= 100 else round(sorted_durations[-1], 1)
                ),
                "min_duration_ms": round(sorted_durations[0], 1),
                "max_duration_ms": round(sorted_durations[-1], 1),
                "recent_errors": bucket["errors"][-3:],
            }

        return result

    def get_recent_tool_calls(self, n: int = 20) -> list[dict]:
        records = self._tool_records[-n:]
        return [
            {
                "tool_name": r.tool_name,
                "trace_id": r.trace_id,
                "agent_name": r.agent_name,
                "session_id": r.session_id,
                "success": r.success,
                "duration_ms": round(r.duration_ms, 2),
                "error": r.error,
                "retry_count": r.retry_count,
                "timestamp": r.timestamp,
            }
            for r in reversed(records)
        ]

    def _get_filtered_records(self, tool_name: str | None,
                              time_window_seconds: float | None) -> list[ToolCallRecord]:
        records = self._tool_records
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        if time_window_seconds:
            cutoff = time.time() - time_window_seconds
            records = [r for r in records if r.timestamp >= cutoff]
        return records

    def get_trace(self, trace_id: str) -> list[dict]:
        spans = self._traces.get(trace_id, [])
        if not spans and self._persist_dir:
            spans = self._load_trace(trace_id)
        return [self._span_to_dict(s) for s in sorted(spans, key=lambda s: s.start_time)]

    def get_recent_traces(self, n: int = 20) -> list[dict]:
        traces = []
        sorted_ids = sorted(
            self._traces.keys(),
            key=lambda k: self._traces[k][0].start_time if self._traces[k] else 0,
            reverse=True,
        )
        for tid in sorted_ids[:n]:
            spans = self._traces[tid]
            if spans:
                sorted_spans = sorted(spans, key=lambda s: s.start_time)
                root = next((s for s in sorted_spans if s.parent_id is None), sorted_spans[0])
                traces.append({
                    "trace_id": tid,
                    "total_duration_ms": round(root.duration_ms, 1),
                    "span_count": len(spans),
                    "root_agent": root.agent_name,
                    "status": "error" if any(s.status == "error" for s in spans) else "success",
                    "start_time": root.start_time,
                })
        return traces

    def _persist_trace(self, trace_id: str):
        if not self._persist_dir:
            return
        spans = self._traces.get(trace_id, [])
        if not spans:
            return
        data = {
            "trace_id": trace_id,
            "spans": [self._span_to_dict(s) for s in sorted(spans, key=lambda s: s.start_time)],
            "saved_at": time.time(),
        }
        filepath = self._persist_dir / f"{trace_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_trace(self, trace_id: str) -> list[Span]:
        if not self._persist_dir:
            return []
        filepath = self._persist_dir / f"{trace_id}.json"
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        spans = []
        for d in data.get("spans", []):
            s = Span(
                span_id=d["span_id"], parent_id=d["parent_id"], trace_id=d["trace_id"],
                agent_name=d["agent_name"], method=d["method"], start_time=0,
                end_time=0, status=d.get("status", "success"), error=d.get("error"),
            )
            s.prompt_tokens = d.get("prompt_tokens", 0)
            s.completion_tokens = d.get("completion_tokens", 0)
            s.total_tokens = d.get("total_tokens", 0)
            spans.append(s)
        return spans

    @staticmethod
    def _span_to_dict(s: Span) -> dict:
        return {
            "span_id": s.span_id,
            "parent_id": s.parent_id,
            "trace_id": s.trace_id,
            "agent_name": s.agent_name,
            "method": s.method,
            "duration_ms": round(s.duration_ms, 2),
            "status": s.status,
            "error": s.error,
            "start_offset_ms": 0,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.total_tokens,
        }


_collector: SpanCollector | None = None


def get_collector() -> SpanCollector:
    global _collector
    if _collector is None:
        _collector = SpanCollector()
    return _collector


def init_collector(max_traces: int = 100, persist_dir: str | None = None) -> SpanCollector:
    global _collector
    _collector = SpanCollector(max_traces=max_traces, persist_dir=persist_dir)
    return _collector