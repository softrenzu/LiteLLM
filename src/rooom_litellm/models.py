from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelSpec:
    name: str
    litellm_model: str
    quality: float = 0.70
    input_cost_per_million: float = 1.0
    output_cost_per_million: float = 3.0
    expected_latency_ms: float = 1500.0
    regions: tuple[str, ...] = ("global",)
    data_classes: tuple[str, ...] = ("public", "internal")
    capabilities: tuple[str, ...] = ("chat",)
    on_prem: bool = False
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouterWeights:
    quality: float = 0.40
    cost: float = 0.20
    latency: float = 0.20
    reliability: float = 0.15
    complexity_fit: float = 0.05

    def normalized(self) -> "RouterWeights":
        total = self.quality + self.cost + self.latency + self.reliability + self.complexity_fit
        if total <= 0:
            raise ValueError("router weights must sum to > 0")
        return RouterWeights(
            quality=self.quality / total,
            cost=self.cost / total,
            latency=self.latency / total,
            reliability=self.reliability / total,
            complexity_fit=self.complexity_fit / total,
        )


@dataclass(slots=True)
class RouterConfig:
    models: list[ModelSpec]
    weights: RouterWeights = field(default_factory=RouterWeights)
    exploration_rate: float = 0.03
    quality_decay: float = 0.97
    ewma_alpha: float = 0.20
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: int = 60
    shadow_rate: float = 0.0
    max_delegation_depth: int = 6
    default_agent_budget_usd: float = 10.0
    privacy_auto_detect: bool = True


@dataclass(slots=True)
class RequestContext:
    messages: list[dict[str, Any]]
    requested_model: str | None = None
    tools: list[dict[str, Any]] | None = None
    data_class: str | None = None
    required_region: str | None = None
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    agent_id: str | None = None
    root_agent_id: str | None = None
    delegation_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
