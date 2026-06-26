"""
Agent 注册表 — 解耦 Agent 与 Supervisor 的硬编码绑定。
每个 Agent 自声明其处理的意图和优先级，Supervisor 通过注册表动态发现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEntry:
    name: str
    agent: Any
    intents: list[str] = field(default_factory=list)
    priority: int = 0
    always_run: bool = False


class AgentRegistry:

    def __init__(self):
        self._agents: dict[str, AgentEntry] = {}

    def register(
        self,
        name: str,
        agent: Any,
        intents: list[str] | None = None,
        priority: int = 0,
        always_run: bool = False,
    ) -> None:
        self._agents[name] = AgentEntry(
            name=name,
            agent=agent,
            intents=intents or [],
            priority=priority,
            always_run=always_run,
        )

    def get_agents_for_intent(self, intent: str) -> list[tuple[str, Any]]:
        candidates = []
        for entry in self._agents.values():
            if entry.always_run or intent in entry.intents:
                candidates.append(entry)

        candidates.sort(key=lambda e: e.priority)
        return [(e.name, e.agent) for e in candidates]

    def get_agent(self, name: str) -> Any | None:
        entry = self._agents.get(name)
        return entry.agent if entry else None

    def get_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def get_all(self) -> list[AgentEntry]:
        return list(self._agents.values())

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)