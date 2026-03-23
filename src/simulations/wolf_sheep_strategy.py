from typing import Any

import mesa

from src.agents.strategy import HeuristicStrategy
from src.simulations.wolf_sheep import GrassPatch, SheepAgent, WolfAgent
from src.utils.decision_logger import DecisionLogger, make_serializable_context


class WolfSheepHeuristicStrategy(HeuristicStrategy):

    def __init__(
        self,
        model: mesa.Model,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.model = model
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        if context["agent_type"] == "sheep":
            action = self._decide_sheep(context)
        else:
            action = self._decide_wolf(context)

        if self.logger is not None:
            self.logger.log_decision(
                agent_id=context.get("agent_id", 0),
                agent_type=context["agent_type"],
                context=make_serializable_context(context),
                action=action,
            )

        return action

    def _decide_sheep(self, context: dict[str, Any]) -> dict[str, Any]:
        cell = context["cell"]
        neighborhood = list(cell.neighborhood)

        safe_cells = [
            c for c in neighborhood
            if not any(isinstance(a, WolfAgent) for a in c.agents)
        ]

        if not safe_cells:
            target_cells = neighborhood
        else:
            cells_with_grass = [
                c for c in safe_cells
                if any(
                    isinstance(a, GrassPatch) and a.fully_grown
                    for a in c.agents
                )
            ]
            target_cells = cells_with_grass if cells_with_grass else safe_cells

        target = self.model.random.choice(target_cells)

        return {
            "move": target.coordinate,
            "eat": context["grass_available"],
            "reproduce": True,
        }

    def _decide_wolf(self, context: dict[str, Any]) -> dict[str, Any]:
        cell = context["cell"]
        neighborhood = list(cell.neighborhood)

        cells_with_sheep = [
            c for c in neighborhood
            if any(isinstance(a, SheepAgent) for a in c.agents)
        ]

        target_cells = cells_with_sheep if cells_with_sheep else neighborhood
        target = self.model.random.choice(target_cells)

        return {
            "move": target.coordinate,
            "eat": True,
            "reproduce": True,
        }
