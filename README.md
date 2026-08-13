# RooomGateway — Enterprise LLM API Gateway

Version: `0.3.0`

RooomGateway is a source-available OpenAI-compatible gateway for enterprise LLM access. It adds policy, privacy, reliability, budget, and explainable routing controls while using LiteLLM as one provider-integration layer.

> RooomGateway is not the official LiteLLM project and is not affiliated with BerriAI. The upstream `litellm` package remains governed by its own license.

## Core features

- Latency-aware adaptive routing using live EWMA latency
- Reliability-aware routing with circuit breaking
- Feedback-driven quality learning with history decay
- Privacy and data-boundary routing for restricted traffic
- Root-agent budgets across delegated agent trees
- Capability contracts for tools, vision, and reasoning
- Shadow evaluation against the next-best eligible model
- Explainable routing metadata
- Per-request latency, cost, region, and data-class controls
- OpenAI-compatible request surface

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=...
uvicorn rooom_litellm.app:app --host 0.0.0.0 --port 4000
```

The internal `rooom_litellm` import path is retained in the `0.3.x` line for compatibility; the product and distribution name are RooomGateway / `rooom-gateway`.

## Licensing and enterprise support

Starting with version `0.3.0`, ROOOMTECH-authored code is available under the terms described in `LICENSE`: PolyForm Noncommercial License 1.0.0 for uses permitted by that license, or a separate paid ROOOMTECH Commercial Software License for business/commercial-purpose uses outside those permissions.

Commercial license agreements, maintenance, technical support, implementation, integration, upgrades, security support, SLA options, private builds, and custom development are available.

Contact: `support@rooomtech.com`

Earlier releases retain their published license terms. The upstream LiteLLM package and all third-party software retain their own licenses.
