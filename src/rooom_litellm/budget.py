from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


class BudgetExceeded(RuntimeError):
    pass


class DelegationDepthExceeded(RuntimeError):
    pass


@dataclass
class AgentBudgetLedger:
    default_budget_usd: float = 10.0
    max_delegation_depth: int = 6
    budgets: dict[str, float] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def set_budget(self, agent_id: str, budget_usd: float) -> None:
        if budget_usd < 0:
            raise ValueError("budget must be >= 0")
        with self._lock:
            self.budgets[agent_id] = budget_usd

    def available(self, root_agent_id: str) -> float:
        with self._lock:
            budget = self.budgets.get(root_agent_id, self.default_budget_usd)
            return max(0.0, budget - self.spent.get(root_agent_id, 0.0))

    def authorize(self, root_agent_id: str, estimated_cost_usd: float, delegation_depth: int) -> None:
        if delegation_depth > self.max_delegation_depth:
            raise DelegationDepthExceeded(
                f"delegation depth {delegation_depth} exceeds limit {self.max_delegation_depth}"
            )
        if estimated_cost_usd > self.available(root_agent_id):
            raise BudgetExceeded(
                f"root agent {root_agent_id!r} has ${self.available(root_agent_id):.6f} remaining"
            )

    def charge(self, root_agent_id: str, amount_usd: float) -> None:
        if amount_usd < 0:
            raise ValueError("charge must be >= 0")
        with self._lock:
            self.spent[root_agent_id] = self.spent.get(root_agent_id, 0.0) + amount_usd
