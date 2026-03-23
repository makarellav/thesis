from typing import Any

import mesa

from src.agents.strategy import DecisionStrategy


class BaseAgent(mesa.Agent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy

    def step(self) -> None:
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        return {
            "agent_id": self.unique_id,
            "model": self.model,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        pass
