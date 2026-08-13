from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from .budget import AgentBudgetLedger
from .metrics import RuntimeStore
from .models import ModelSpec, RequestContext, RouterConfig
from .policy import PolicyResult, evaluate_policy, flatten_text, model_allowed


@dataclass(slots=True)
class RouteDecision:
    model: ModelSpec
    score: float
    reason: dict[str, float | str | bool]
    shadow_model: ModelSpec | None = None
    estimated_cost_usd: float = 0.0
    data_class: str = "internal"


def _rough_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 3.5))


def _complexity(ctx: RequestContext) -> float:
    text = flatten_text(ctx.messages).lower()
    token_est = _rough_tokens(text)
    score = min(0.55, token_est / 5000)
    hard_markers = (
        "prove", "derive", "architecture", "root cause", "tradeoff", "optimize",
        "証明", "設計", "原因", "比較", "最適化", "アーキテクチャ",
    )
    score += min(0.25, sum(marker in text for marker in hard_markers) * 0.05)
    if ctx.tools:
        score += 0.12
    if ctx.metadata.get("reasoning"):
        score += 0.15
    return min(1.0, score)


def _cost_estimate(model: ModelSpec, ctx: RequestContext) -> float:
    prompt_tokens = _rough_tokens(flatten_text(ctx.messages))
    expected_output_tokens = int(ctx.metadata.get("expected_output_tokens", 600))
    return (
        prompt_tokens * model.input_cost_per_million
        + expected_output_tokens * model.output_cost_per_million
    ) / 1_000_000


class NoEligibleModel(RuntimeError):
    pass


class AdaptiveRouter:
    """Multi-objective, privacy-aware, feedback-learning router."""

    def __init__(self, config: RouterConfig, rng: random.Random | None = None):
        if not config.models:
            raise ValueError("at least one model is required")
        self.config = config
        self.weights = config.weights.normalized()
        self.rng = rng or random.Random()
        self.runtime = RuntimeStore(
            config.ewma_alpha,
            config.circuit_breaker_failures,
            config.circuit_breaker_cooldown_s,
        )
        self.ledger = AgentBudgetLedger(
            default_budget_usd=config.default_agent_budget_usd,
            max_delegation_depth=config.max_delegation_depth,
        )
        for model in config.models:
            self.runtime.ensure(model.name, model.expected_latency_ms, model.quality)

    def eligible_models(self, ctx: RequestContext, policy: PolicyResult) -> list[ModelSpec]:
        models = [m for m in self.config.models if model_allowed(m, ctx, policy)]
        models = [m for m in models if self.runtime.is_available(m.name, m.expected_latency_ms, m.quality)]
        if ctx.requested_model and ctx.requested_model not in {"auto", "smart"}:
            models = [m for m in models if m.name == ctx.requested_model or m.litellm_model == ctx.requested_model]
        return models

    def _score_all(self, models: Iterable[ModelSpec], ctx: RequestContext) -> list[tuple[ModelSpec, float, dict[str, float]]]:
        models = list(models)
        if not models:
            return []
        runtimes = self.runtime.snapshot()
        complexity = _complexity(ctx)
        costs = {m.name: _cost_estimate(m, ctx) for m in models}
        max_cost = max(costs.values()) or 1.0
        latencies = {m.name: float(runtimes[m.name]["latency_ms"]) for m in models}
        max_latency = max(latencies.values()) or 1.0

        rows: list[tuple[ModelSpec, float, dict[str, float]]] = []
        for m in models:
            rt = runtimes[m.name]
            quality = float(rt["quality"])
            reliability = float(rt["reliability"])
            cost_score = 1.0 - min(1.0, costs[m.name] / max_cost)
            latency_score = 1.0 - min(1.0, latencies[m.name] / max_latency)
            complexity_fit = 1.0 - abs(quality - complexity)
            if ctx.max_latency_ms:
                latency_score *= 1.0 if latencies[m.name] <= ctx.max_latency_ms else 0.1
            score = (
                self.weights.quality * quality
                + self.weights.cost * cost_score
                + self.weights.latency * latency_score
                + self.weights.reliability * reliability
                + self.weights.complexity_fit * complexity_fit
            )
            rows.append((m, score, {
                "quality": quality,
                "cost": cost_score,
                "latency": latency_score,
                "reliability": reliability,
                "complexity_fit": complexity_fit,
                "complexity": complexity,
            }))
        return sorted(rows, key=lambda x: x[1], reverse=True)

    def route(self, ctx: RequestContext) -> RouteDecision:
        policy = evaluate_policy(ctx, self.config.privacy_auto_detect)
        if policy.rejected_reason:
            raise NoEligibleModel(policy.rejected_reason)
        models = self.eligible_models(ctx, policy)
        scored = self._score_all(models, ctx)
        if not scored:
            raise NoEligibleModel(
                f"no model satisfies data_class={policy.data_class}, region={ctx.required_region}, capabilities={sorted(policy.required_capabilities)}"
            )

        if len(scored) > 1 and self.rng.random() < self.config.exploration_rate:
            chosen = self.rng.choice(scored[1:])
            exploration = True
        else:
            chosen = scored[0]
            exploration = False

        model, score, components = chosen
        estimated_cost = _cost_estimate(model, ctx)
        if ctx.max_cost_usd is not None and estimated_cost > ctx.max_cost_usd:
            cheaper = [row for row in scored if _cost_estimate(row[0], ctx) <= ctx.max_cost_usd]
            if not cheaper:
                raise NoEligibleModel(f"all eligible models exceed max_cost_usd={ctx.max_cost_usd}")
            model, score, components = cheaper[0]
            estimated_cost = _cost_estimate(model, ctx)

        root_agent = ctx.root_agent_id or ctx.agent_id
        if root_agent:
            self.ledger.authorize(root_agent, estimated_cost, ctx.delegation_depth)

        shadow = None
        if len(scored) > 1 and self.config.shadow_rate > 0 and self.rng.random() < self.config.shadow_rate:
            shadow = next((row[0] for row in scored if row[0].name != model.name), None)

        reason: dict[str, float | str | bool] = {**components, "exploration": exploration}
        return RouteDecision(
            model=model,
            score=score,
            reason=reason,
            shadow_model=shadow,
            estimated_cost_usd=estimated_cost,
            data_class=policy.data_class,
        )

    def record_success(self, model_name: str, latency_ms: float) -> None:
        model = self._find(model_name)
        self.runtime.record_success(model.name, latency_ms, model.quality)

    def record_failure(self, model_name: str, latency_ms: float) -> None:
        model = self._find(model_name)
        self.runtime.record_failure(model.name, latency_ms, model.quality)

    def record_feedback(self, model_name: str, score: float) -> None:
        model = self._find(model_name)
        self.runtime.feedback(model.name, score, model.quality, self.config.quality_decay)

    def charge(self, root_agent_id: str | None, amount_usd: float) -> None:
        if root_agent_id:
            self.ledger.charge(root_agent_id, amount_usd)

    def _find(self, name: str) -> ModelSpec:
        for model in self.config.models:
            if model.name == name or model.litellm_model == name:
                return model
        raise KeyError(name)
