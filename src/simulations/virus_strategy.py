from typing import Any

import mesa

from src.agents.strategy import HeuristicStrategy
from src.utils.decision_logger import DecisionLogger, make_serializable_context


class VirusHeuristicStrategy(HeuristicStrategy):

    def __init__(
        self,
        model: mesa.Model,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.model = model
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        cell = context["cell"]
        new_cell = cell.neighborhood.select_random_cell()
        new_pos = new_cell.coordinate

        action = {"move": new_pos, "interact": True}

        if self.logger is not None:
            self.logger.log_decision(
                agent_id=context.get("agent_id", 0),
                agent_type=context.get("state", "unknown"),
                context=make_serializable_context(context),
                action=action,
            )

        return action
