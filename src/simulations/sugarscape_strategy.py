"""Sugarscape-specific decision strategies.

This module implements concrete strategies for Sugarscape agents,
including the classic welfare-maximizing movement heuristic.
"""

from typing import Any

from src.agents.strategy import HeuristicStrategy


class SugarscapeHeuristicStrategy(HeuristicStrategy):
    """Algorithmic strategy for Sugarscape agent movement.

    Implements the classic Sugarscape movement rule:
    1. Look at all cells within vision range
    2. Calculate welfare (total resources) for each cell
    3. Move to the unoccupied cell with highest welfare
    4. If tied, move to the closest one
    """

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        """Decide where to move based on resource availability.

        Args:
            context: Contains position, vision, neighbors, and grid reference.

        Returns:
            Action dictionary with 'move' key containing target position.
        """
        current_pos = context["position"]
        neighbors = context["neighbors"]
        grid = context["grid"]

        sugar_layer = grid.sugar
        spice_layer = grid.spice

        best_pos = current_pos
        best_welfare = self._calculate_welfare(current_pos, sugar_layer, spice_layer)
        best_distance = float("inf")

        for neighbor_cell in neighbors:
            neighbor_pos = neighbor_cell.coordinate

            if len(neighbor_cell.agents) > 0:
                continue

            welfare = self._calculate_welfare(neighbor_pos, sugar_layer, spice_layer)
            distance = self._manhattan_distance(current_pos, neighbor_pos)

            if welfare > best_welfare or (
                welfare == best_welfare and distance < best_distance
            ):
                best_pos = neighbor_pos
                best_welfare = welfare
                best_distance = distance

        return {"move": best_pos}

    def _calculate_welfare(
        self,
        pos: tuple[int, int],
        sugar_layer: Any,
        spice_layer: Any,
    ) -> float:
        """Calculate welfare (total resources) at a position.

        Args:
            pos: Grid position to evaluate.
            sugar_layer: Sugar resource layer.
            spice_layer: Spice resource layer.

        Returns:
            Total welfare (sugar + spice) at the position.
        """
        return float(sugar_layer.data[pos] + spice_layer.data[pos])

    def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions.

        Args:
            pos1: First position.
            pos2: Second position.

        Returns:
            Manhattan distance.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
