from typing import Any

from src.agents.strategy import HeuristicStrategy
from src.utils.decision_logger import DecisionLogger, make_serializable_context


class SugarscapeHeuristicStrategy(HeuristicStrategy):

    def __init__(
        self,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        current_pos = context["position"]
        neighbors = context["neighbors"]
        grid = context["grid"]

        sugar_layer = grid.sugar
        spice_layer = grid.spice

        best_pos = current_pos
        best_welfare = self._calculate_welfare(
            current_pos, sugar_layer, spice_layer
        )
        best_distance = float("inf")

        for neighbor_cell in neighbors:
            neighbor_pos = neighbor_cell.coordinate

            if len(neighbor_cell.agents) > 0:
                continue

            welfare = self._calculate_welfare(
                neighbor_pos, sugar_layer, spice_layer
            )
            distance = self._manhattan_distance(current_pos, neighbor_pos)

            if welfare > best_welfare or (
                welfare == best_welfare and distance < best_distance
            ):
                best_pos = neighbor_pos
                best_welfare = welfare
                best_distance = distance

        action = {"move": best_pos}

        if self.logger is not None:
            self.logger.log_decision(
                agent_id=context.get("agent_id", 0),
                agent_type="sugar_agent",
                context=make_serializable_context(context),
                action=action,
            )

        return action

    def _calculate_welfare(
        self,
        pos: tuple[int, int],
        sugar_layer: Any,
        spice_layer: Any,
    ) -> float:
        return float(sugar_layer.data[pos] + spice_layer.data[pos])

    def _manhattan_distance(
        self, pos1: tuple[int, int], pos2: tuple[int, int]
    ) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
