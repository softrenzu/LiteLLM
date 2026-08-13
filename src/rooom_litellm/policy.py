from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ModelSpec, RequestContext


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?81[- ]?)?0\d{1,4}[- ]?\d{1,4}[- ]?\d{3,4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
_JP_MY_NUMBER = re.compile(r"(?<!\d)\d{12}(?!\d)")
_SECRET = re.compile(r"(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*\S+", re.I)


@dataclass(slots=True)
class PolicyResult:
    data_class: str
    required_capabilities: set[str]
    rejected_reason: str | None = None


def flatten_text(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "\n".join(parts)


def classify_data(messages: list[dict]) -> str:
    text = flatten_text(messages)
    if _SECRET.search(text) or _CARD.search(text) or _JP_MY_NUMBER.search(text):
        return "restricted"
    if _EMAIL.search(text) or _PHONE.search(text):
        return "confidential"
    return "internal"


def infer_capabilities(ctx: RequestContext) -> set[str]:
    caps = {"chat"}
    if ctx.tools:
        caps.add("tools")
    for msg in ctx.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image_url", "input_image"}:
                    caps.add("vision")
    if ctx.metadata.get("reasoning"):
        caps.add("reasoning")
    return caps


def evaluate_policy(ctx: RequestContext, auto_detect: bool = True) -> PolicyResult:
    data_class = ctx.data_class or (classify_data(ctx.messages) if auto_detect else "internal")
    if ctx.delegation_depth < 0:
        return PolicyResult(data_class, set(), "delegation_depth must be >= 0")
    return PolicyResult(data_class, infer_capabilities(ctx))


def model_allowed(model: ModelSpec, ctx: RequestContext, policy: PolicyResult) -> bool:
    if not model.enabled:
        return False
    if policy.data_class not in model.data_classes:
        return False
    if ctx.required_region and ctx.required_region not in model.regions:
        return False
    if not policy.required_capabilities.issubset(set(model.capabilities)):
        return False
    if policy.data_class == "restricted" and not model.on_prem:
        return False
    return True
