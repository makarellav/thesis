from pathlib import Path
from typing import Any

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent, OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer

from src.agents.strategy import DecisionStrategy
from src.utils.metrics import calculate_gini


class SugarAgent(CellAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
        cell: Any,
        sugar: int,
        spice: int,
        metabolism_sugar: int,
        metabolism_spice: int,
        vision: int,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.age = 0
        self.total_moves = 0
        self.suboptimal_moves = 0

    def step(self) -> None:
        self.age += 1
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        current_pos = self.cell.coordinate

        neighbors = self.cell.get_neighborhood(
            radius=self.vision,
        )

        return {
            "agent_id": self.unique_id,
            "position": current_pos,
            "sugar": self.sugar,
            "spice": self.spice,
            "metabolism_sugar": self.metabolism_sugar,
            "metabolism_spice": self.metabolism_spice,
            "vision": self.vision,
            "neighbors": neighbors,
            "grid": self.model.grid,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        if "move" in action:
            new_pos = action["move"]
            current_pos = self.cell.coordinate

            if new_pos != current_pos:
                self.total_moves += 1

                if not self._is_optimal_move(new_pos):
                    self.suboptimal_moves += 1

            new_cell = self.model.grid[new_pos]
            self.cell = new_cell

        self.eat()

    def eat(self) -> None:
        sugar_patch = self.model.grid.sugar
        spice_patch = self.model.grid.spice

        cell_pos = self.cell.coordinate

        self.sugar += sugar_patch.data[cell_pos]
        self.spice += spice_patch.data[cell_pos]

        sugar_patch.data[cell_pos] = 0
        spice_patch.data[cell_pos] = 0

        self.sugar -= self.metabolism_sugar
        self.spice -= self.metabolism_spice

    def is_starved(self) -> bool:
        return self.sugar <= 0 or self.spice <= 0

    def _is_optimal_move(self, chosen_pos: tuple[int, int]) -> bool:
        neighbors = self.cell.get_neighborhood(radius=self.vision)

        sugar_layer = self.model.grid.sugar
        spice_layer = self.model.grid.spice

        chosen_welfare = float(
            sugar_layer.data[chosen_pos] + spice_layer.data[chosen_pos]
        )

        best_welfare = chosen_welfare

        for neighbor_cell in neighbors:
            if len(neighbor_cell.agents) > 0:
                continue

            neighbor_pos = neighbor_cell.coordinate
            welfare = float(
                sugar_layer.data[neighbor_pos] + spice_layer.data[neighbor_pos]
            )

            if welfare > best_welfare:
                best_welfare = welfare

        return chosen_welfare >= best_welfare


class SugarscapeModel(mesa.Model):  # type: ignore[misc]

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        initial_population: int = 100,
        endowment_min: int = 25,
        endowment_max: int = 50,
        metabolism_min: int = 1,
        metabolism_max: int = 5,
        vision_min: int = 1,
        vision_max: int = 5,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height

        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height),
            torus=False,
            random=self.random,
        )

        sugar_map_path = Path(__file__).parent / "sugar-map.txt"
        self.sugar_distribution = np.loadtxt(sugar_map_path)

        self.spice_distribution = np.flip(self.sugar_distribution, 1)

        sugar_layer = PropertyLayer(
            "sugar", (self.width, self.height), default_value=0, dtype=int
        )
        sugar_layer.data = self.sugar_distribution.copy().astype(int)
        self.grid.add_property_layer(sugar_layer)

        spice_layer = PropertyLayer(
            "spice", (self.width, self.height), default_value=0, dtype=int
        )
        spice_layer.data = self.spice_distribution.copy().astype(int)
        self.grid.add_property_layer(spice_layer)

        self.datacollector = DataCollector(
            model_reporters={
                "Agent Count": lambda m: len(m.agents),
                "Gini Coefficient": lambda m: calculate_gini(
                    [
                        a.sugar + a.spice
                        for a in m.agents
                        if isinstance(a, SugarAgent)
                    ]
                ),
                "Average Lifespan": lambda m: (
                    sum(a.age for a in m.agents if isinstance(a, SugarAgent))
                    / len(m.agents)
                    if len(m.agents) > 0
                    else 0
                ),
                "Average Wealth": lambda m: (
                    sum(
                        a.sugar + a.spice
                        for a in m.agents
                        if isinstance(a, SugarAgent)
                    )
                    / len(m.agents)
                    if len(m.agents) > 0
                    else 0
                ),
                "Suboptimal Move Rate": lambda m: (
                    sum(
                        a.suboptimal_moves
                        for a in m.agents
                        if isinstance(a, SugarAgent)
                    )
                    / sum(
                        a.total_moves
                        for a in m.agents
                        if isinstance(a, SugarAgent)
                    )
                    if sum(
                        a.total_moves
                        for a in m.agents
                        if isinstance(a, SugarAgent)
                    )
                    > 0
                    else 0
                ),
            },
            agent_reporters={
                "Sugar": "sugar",
                "Spice": "spice",
                "Age": "age",
                "Total Moves": "total_moves",
                "Suboptimal Moves": "suboptimal_moves",
            },
        )

    def step(self) -> None:
        sugar_layer = self.grid.sugar
        spice_layer = self.grid.spice

        sugar_layer.data = np.minimum(
            sugar_layer.data + 1,
            self.sugar_distribution,
        ).astype(int)
        spice_layer.data = np.minimum(
            spice_layer.data + 1,
            self.spice_distribution,
        ).astype(int)

        self.agents.shuffle_do("step")

        agents_to_remove = [
            agent
            for agent in self.agents
            if isinstance(agent, SugarAgent) and agent.is_starved()
        ]
        for agent in agents_to_remove:
            self.agents.remove(agent)

        self.datacollector.collect(self)
