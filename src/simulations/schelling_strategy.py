from typing import Any

from src.agents.strategy import HeuristicStrategy
from src.utils.decision_logger import DecisionLogger, make_serializable_context


class SchellingHeuristicStrategy(HeuristicStrategy):

    def __init__(
        self,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        similar_fraction = context["similar_fraction"]
        homophily = context["homophily"]
        total_neighbors = context["total_neighbors"]

        if total_neighbors == 0 or similar_fraction >= homophily:
            action = {"move": False}
        else:
            action = {"move": True}

        if self.logger is not None:
            self.logger.log_decision(
                agent_id=context.get("agent_id", 0),
                agent_type=str(context.get("agent_type", "unknown")),
                context=make_serializable_context(context),
                action=action,
            )

        return action
