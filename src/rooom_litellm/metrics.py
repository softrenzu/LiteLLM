from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock


@dataclass(slots=True)
class ModelRuntime:
    latency_ms: float
    reliability: float = 1.0
    quality: float = 0.70
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    feedback_count: int = 0


class RuntimeStore:
    def __init__(self, ewma_alpha: float, breaker_failures: int, breaker_cooldown_s: int):
        self.alpha = ewma_alpha
        self.breaker_failures = breaker_failures
        self.breaker_cooldown_s = breaker_cooldown_s
        self._items: dict[str, ModelRuntime] = {}
        self._lock = RLock()

    def ensure(self, name: str, latency_ms: float, quality: float) -> ModelRuntime:
        with self._lock:
            return self._items.setdefault(name, ModelRuntime(latency_ms=latency_ms, quality=quality))

    def is_available(self, name: str, latency_ms: float, quality: float) -> bool:
        item = self.ensure(name, latency_ms, quality)
        return time.time() >= item.circuit_open_until

    def record_success(self, name: str, latency_ms: float, quality: float) -> None:
        with self._lock:
            item = self.ensure(name, latency_ms, quality)
            item.latency_ms = self.alpha * latency_ms + (1 - self.alpha) * item.latency_ms
            item.success_count += 1
            item.consecutive_failures = 0
            total = item.success_count + item.failure_count
            item.reliability = item.success_count / total if total else 1.0

    def record_failure(self, name: str, latency_ms: float, quality: float) -> None:
        with self._lock:
            item = self.ensure(name, latency_ms, quality)
            item.failure_count += 1
            item.consecutive_failures += 1
            total = item.success_count + item.failure_count
            item.reliability = item.success_count / total if total else 0.0
            if item.consecutive_failures >= self.breaker_failures:
                item.circuit_open_until = time.time() + self.breaker_cooldown_s

    def feedback(self, name: str, score: float, quality: float, decay: float = 0.97) -> None:
        score = min(1.0, max(0.0, score))
        with self._lock:
            item = self.ensure(name, 1500.0, quality)
            effective_alpha = max(0.02, 1.0 - decay)
            item.quality = effective_alpha * score + (1 - effective_alpha) * item.quality
            item.feedback_count += 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return {
                name: {
                    "latency_ms": round(v.latency_ms, 2),
                    "reliability": round(v.reliability, 4),
                    "quality": round(v.quality, 4),
                    "success_count": v.success_count,
                    "failure_count": v.failure_count,
                    "feedback_count": v.feedback_count,
                    "circuit_open_until": v.circuit_open_until,
                }
                for name, v in self._items.items()
            }
