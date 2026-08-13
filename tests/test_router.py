import random

import pytest

from rooom_litellm.budget import BudgetExceeded, DelegationDepthExceeded
from rooom_litellm.models import ModelSpec, RequestContext, RouterConfig, RouterWeights
from rooom_litellm.policy import classify_data
from rooom_litellm.router import AdaptiveRouter, NoEligibleModel


def make_router(**overrides):
    models = [
        ModelSpec(
            name="cheap", litellm_model="openai/cheap", quality=0.65,
            input_cost_per_million=0.2, output_cost_per_million=0.8,
            expected_latency_ms=300, regions=("jp",),
            data_classes=("public", "internal"), capabilities=("chat", "tools")
        ),
        ModelSpec(
            name="strong", litellm_model="openai/strong", quality=0.95,
            input_cost_per_million=2.0, output_cost_per_million=8.0,
            expected_latency_ms=1200, regions=("jp",),
            data_classes=("public", "internal", "confidential"), capabilities=("chat", "tools", "reasoning")
        ),
        ModelSpec(
            name="private", litellm_model="ollama/private", quality=0.80,
            input_cost_per_million=0.1, output_cost_per_million=0.2,
            expected_latency_ms=800, regions=("jp",),
            data_classes=("public", "internal", "confidential", "restricted"),
            capabilities=("chat", "tools"), on_prem=True,
        ),
    ]
    cfg = RouterConfig(models=models, exploration_rate=0.0, **overrides)
    return AdaptiveRouter(cfg, rng=random.Random(1))


def test_restricted_data_forces_on_prem():
    r = make_router()
    ctx = RequestContext(messages=[{"role": "user", "content": "api_key=supersecret"}], required_region="jp")
    decision = r.route(ctx)
    assert decision.model.name == "private"
    assert decision.data_class == "restricted"


def test_confidential_detects_email():
    assert classify_data([{"role": "user", "content": "mail me at a@example.com"}]) == "confidential"


def test_capability_filter_for_reasoning():
    r = make_router()
    ctx = RequestContext(
        messages=[{"role": "user", "content": "derive a system architecture"}],
        required_region="jp", metadata={"reasoning": True},
    )
    assert r.route(ctx).model.name == "strong"


def test_budget_is_root_agent_scoped():
    r = make_router(default_agent_budget_usd=0.000001)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "hello"}],
        agent_id="child", root_agent_id="root", delegation_depth=1,
    )
    with pytest.raises(BudgetExceeded):
        r.route(ctx)


def test_delegation_depth_limit():
    r = make_router(max_delegation_depth=2)
    ctx = RequestContext(
        messages=[{"role": "user", "content": "hello"}],
        root_agent_id="root", delegation_depth=3,
    )
    with pytest.raises(DelegationDepthExceeded):
        r.route(ctx)


def test_circuit_breaker_removes_failed_model():
    r = make_router(circuit_breaker_failures=2, circuit_breaker_cooldown_s=3600,
                    weights=RouterWeights(quality=1, cost=0, latency=0, reliability=0, complexity_fit=0))
    ctx = RequestContext(messages=[{"role": "user", "content": "normal question"}], data_class="confidential")
    assert r.route(ctx).model.name == "strong"
    r.record_failure("strong", 2000)
    r.record_failure("strong", 2000)
    assert r.route(ctx).model.name == "private"


def test_feedback_can_shift_quality_over_time():
    r = make_router(quality_decay=0.50,
                    weights=RouterWeights(quality=1, cost=0, latency=0, reliability=0, complexity_fit=0))
    ctx = RequestContext(messages=[{"role": "user", "content": "hello"}], data_class="internal")
    assert r.route(ctx).model.name == "strong"
    for _ in range(4):
        r.record_feedback("strong", 0.0)
        r.record_feedback("private", 1.0)
    assert r.route(ctx).model.name == "private"
