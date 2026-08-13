from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from litellm import acompletion

from .budget import BudgetExceeded, DelegationDepthExceeded
from .config import load_config
from .models import RequestContext
from .router import AdaptiveRouter, NoEligibleModel, RouteDecision

CONFIG_PATH = os.getenv("ROOOM_LITELLM_CONFIG", "examples/config.yaml")
router = AdaptiveRouter(load_config(CONFIG_PATH))
app = FastAPI(title="Rooom LiteLLM+", version="0.1.0")


def _root_agent(ctx: RequestContext) -> str | None:
    return ctx.root_agent_id or ctx.agent_id


def _actual_cost(decision: RouteDecision, response: Any) -> float:
    usage = getattr(response, "usage", None)
    if not usage:
        return decision.estimated_cost_usd
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    model = decision.model
    return (
        prompt * model.input_cost_per_million + completion * model.output_cost_per_million
    ) / 1_000_000


async def _shadow_call(decision: RouteDecision, body: dict[str, Any]) -> None:
    if not decision.shadow_model:
        return
    shadow_body = dict(body)
    shadow_body.pop("model", None)
    shadow_body.pop("router", None)
    started = time.perf_counter()
    try:
        await acompletion(model=decision.shadow_model.litellm_model, **shadow_body)
        router.record_success(decision.shadow_model.name, (time.perf_counter() - started) * 1000)
    except Exception:
        router.record_failure(decision.shadow_model.name, (time.perf_counter() - started) * 1000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/router/models")
def models() -> dict[str, Any]:
    return {
        "models": [m.name for m in router.config.models if m.enabled],
        "runtime": router.runtime.snapshot(),
    }


@app.post("/v1/router/feedback")
def feedback(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body["model"])
    score = float(body["score"])
    router.record_feedback(model, score)
    return {"ok": True, "model": model, "score": max(0.0, min(1.0, score))}


@app.post("/v1/router/budgets/{agent_id}")
def set_budget(agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
    budget = float(body["budget_usd"])
    router.ledger.set_budget(agent_id, budget)
    return {"ok": True, "agent_id": agent_id, "budget_usd": budget}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    router_meta = dict(body.pop("router", {}) or {})
    ctx = RequestContext(
        messages=body.get("messages", []),
        requested_model=body.get("model"),
        tools=body.get("tools"),
        data_class=router_meta.get("data_class") or request.headers.get("x-data-class"),
        required_region=router_meta.get("required_region") or request.headers.get("x-required-region"),
        max_latency_ms=router_meta.get("max_latency_ms"),
        max_cost_usd=router_meta.get("max_cost_usd"),
        agent_id=request.headers.get("x-agent-id") or router_meta.get("agent_id"),
        root_agent_id=request.headers.get("x-root-agent-id") or router_meta.get("root_agent_id"),
        delegation_depth=int(request.headers.get("x-delegation-depth", router_meta.get("delegation_depth", 0))),
        metadata=router_meta,
    )
    try:
        decision = router.route(ctx)
    except (NoEligibleModel, BudgetExceeded, DelegationDepthExceeded) as exc:
        raise HTTPException(status_code=429 if isinstance(exc, BudgetExceeded) else 400, detail=str(exc)) from exc

    upstream = dict(body)
    upstream["model"] = decision.model.litellm_model
    started = time.perf_counter()
    try:
        response = await acompletion(**upstream)
    except Exception as exc:
        router.record_failure(decision.model.name, (time.perf_counter() - started) * 1000)
        raise HTTPException(status_code=502, detail=f"upstream model failed: {type(exc).__name__}") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    router.record_success(decision.model.name, latency_ms)
    cost = _actual_cost(decision, response)
    router.charge(_root_agent(ctx), cost)

    if decision.shadow_model:
        asyncio.create_task(_shadow_call(decision, body))

    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    headers = {
        "x-rooom-model": decision.model.name,
        "x-rooom-score": f"{decision.score:.6f}",
        "x-rooom-data-class": decision.data_class,
        "x-rooom-estimated-cost-usd": f"{decision.estimated_cost_usd:.8f}",
    }
    if decision.shadow_model:
        headers["x-rooom-shadow-model"] = decision.shadow_model.name
    return JSONResponse(content=payload, headers=headers)
