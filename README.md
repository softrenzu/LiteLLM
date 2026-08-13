# Rooom LiteLLM+

A small OpenAI-compatible routing layer built on top of upstream LiteLLM. It keeps LiteLLM's provider abstraction while adding enterprise routing behavior that is useful when cost alone is not enough.

> This repository is not the official BerriAI/LiteLLM project and is not affiliated with BerriAI. It uses `litellm` as a runtime dependency rather than copying upstream source code.

## Why this exists

Upstream LiteLLM already provides a broad production gateway: a unified interface for 100+ LLM providers, OpenAI-compatible endpoints, authentication, virtual keys, spend tracking, rate limits, guardrails, caching, logging, retries/fallbacks and multiple routing strategies. Its current Adaptive Router also performs request-type-aware model selection with a quality/cost bandit.

Rooom LiteLLM+ focuses on additional behavior around that gateway:

| Feature | Rooom LiteLLM+ behavior |
|---|---|
| Latency-aware adaptive routing | Live EWMA latency enters every routing score instead of quality/cost only. |
| Reliability-aware routing | Error rate and a circuit breaker automatically quarantine unstable model endpoints. |
| Drift-aware quality learning | Explicit feedback is decayed so old model quality does not dominate forever. |
| Privacy/data boundary routing | Automatic PII/secret classification plus hard data-class and region filters; `restricted` traffic is on-prem only. |
| Agent delegation budgets | Spend is authorized against the root agent while child agents carry delegation depth. |
| Capability contracts | Models are filtered before scoring for tools, vision and reasoning requirements. |
| Shadow evaluation | A configurable sample can be sent to the next-best eligible model for side-by-side evaluation. |
| Explainable routing | Each decision returns model, score, data classification and cost estimate in response headers. |
| Per-request SLO controls | `max_latency_ms`, `max_cost_usd`, region and data class can be provided per request. |

## Architecture

```text
OpenAI SDK / Agent
        |
        v
POST /v1/chat/completions
        |
        +--> privacy + capability policy
        +--> root-agent budget/delegation check
        +--> candidate filtering
        +--> quality + cost + latency + reliability + complexity score
        +--> circuit breaker / exploration / optional shadow
        |
        v
      LiteLLM
        |
        +--> OpenAI / Anthropic / Gemini / Azure / Bedrock / NVIDIA NIM / Ollama / ...
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=...
uvicorn rooom_litellm.app:app --host 0.0.0.0 --port 4000
```

Or:

```bash
docker build -t rooom-litellm-plus .
docker run --rm -p 4000:4000 -e OPENAI_API_KEY rooom-litellm-plus
```

The example configuration is in `examples/config.yaml`. Replace model names and prices with the endpoints you operate.

## OpenAI-compatible request

```bash
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Agent-ID: planner-2' \
  -H 'X-Root-Agent-ID: travel-agent' \
  -H 'X-Delegation-Depth: 2' \
  -d '{
    "model": "auto",
    "messages": [{"role":"user","content":"Compare the options and explain the tradeoffs"}],
    "router": {
      "required_region": "jp",
      "max_latency_ms": 1500,
      "max_cost_usd": 0.03,
      "reasoning": true
    }
  }'
```

Routing details are exposed as response headers such as `x-rooom-model`, `x-rooom-score`, `x-rooom-data-class` and `x-rooom-estimated-cost-usd`.

## Feedback loop

Feed observed quality back to the router on a 0.0-1.0 scale:

```bash
curl -X POST http://localhost:4000/v1/router/feedback \
  -H 'Content-Type: application/json' \
  -d '{"model":"balanced-cloud","score":0.95}'
```

Unlike a permanently accumulating posterior, older quality history is decayed. This allows routing to adapt when a provider changes a model, quantization, serving stack or capacity profile.

## Privacy rules

Without an explicit `router.data_class`, the gateway performs lightweight deterministic classification:

- secrets, card-like numbers or Japanese My Number-like 12 digit identifiers -> `restricted`
- email/phone-like identifiers -> `confidential`
- otherwise -> `internal`

`restricted` requests can only go to a model configured with `on_prem: true` and matching data-class permissions. This is a routing policy, not a complete DLP system; production deployments should combine it with dedicated DLP/guardrail products where required.

## Agent economics

Requests may contain `X-Agent-ID`, `X-Root-Agent-ID` and `X-Delegation-Depth`. The estimated cost is authorized against the root agent before dispatch and the actual/estimated post-call cost is charged to that same root. This prevents a delegated child-agent tree from bypassing a top-level budget simply by spawning more agents.

For multi-instance production deployments, replace the in-memory ledger/runtime store with Redis or Postgres using the same interfaces.

## Tests

```bash
pip install -e '.[dev]'
pytest -q
```

The tests cover privacy confinement, capability routing, root-agent budgets, delegation depth, circuit breaking and feedback-driven model changes.

## Upstream references

Feature comparison was based on the upstream LiteLLM project and its Adaptive Router documentation inspected in August 2026:

- https://github.com/BerriAI/litellm
- https://docs.litellm.ai/
- https://github.com/BerriAI/litellm/blob/litellm_internal_staging/litellm/router_strategy/adaptive_router/README.md

## License

MIT. The upstream `litellm` package remains governed by its own license.
