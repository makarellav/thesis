from typing import Any

from src.agents.strategy import HeuristicStrategy
from src.utils.decision_logger import DecisionLogger, make_serializable_context


class BoltzmannHeuristicStrategy(HeuristicStrategy):

    def __init__(
        self,
        model: Any,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.model = model
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        cell = context["cell"]
        wealth = context["wealth"]

        new_cell = cell.neighborhood.select_random_cell()
        new_pos = new_cell.coordinate

        give = wealth > 0

        action = {"move": new_pos, "give": give}

        if self.logger is not None:
            self.logger.log_decision(
                agent_id=context.get("agent_id", 0),
                agent_type="wealth_agent",
                context=make_serializable_context(context),
                action=action,
            )

        return action
