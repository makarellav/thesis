"""Sugarscape simulation implementation.

This module implements the Sugarscape G1MT model with Strategy Pattern
architecture. The environment logic (grid, resource growth) is adapted
from Mesa's official example, while agent decision-making is delegated
to strategy instances.

References:
    - Growing Artificial Societies (Epstein & Axtell, 1996)
    - Mesa Sugarscape G1MT example
"""

from pathlib import Path
from typing import Any

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent, OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer

from src.agents.strategy import DecisionStrategy


class SugarAgent(CellAgent):  # type: ignore[misc]
    """Sugarscape agent that gathers resources and trades.

    This agent represents the "body" - managing resource reserves,
    metabolism, vision, and action execution. Decision-making about
    movement and trading is delegated to the strategy.

    Attributes:
        sugar: Current sugar reserves.
        spice: Current spice reserves.
        metabolism_sugar: Sugar consumption per step.
        metabolism_spice: Spice consumption per step.
        vision: Number of cells agent can see in each direction.
    """

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
        """Initialize a Sugarscape agent.

        Args:
            model: The model instance this agent belongs to.
            strategy: Decision strategy for movement and trading.
            cell: The grid cell where the agent is placed.
            sugar: Initial sugar endowment.
            spice: Initial spice endowment.
            metabolism_sugar: Sugar consumption rate per step.
            metabolism_spice: Spice consumption rate per step.
            vision: Vision range in grid cells.
        """
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice
        self.vision = vision

    def step(self) -> None:
        """Execute one time step for this agent.

        Follows the Strategy Pattern: delegates decision-making to the strategy,
        then executes the resulting action.
        """
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        """Gather context for decision-making.

        Returns:
            Dictionary containing agent state and environmental information.
        """
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
        """Execute the decided action.

        Args:
            action: Action dictionary containing 'move' or other commands.
        """
        if "move" in action:
            new_pos = action["move"]
            new_cell = self.model.grid[new_pos]
            self.cell = new_cell

        self.eat()

    def eat(self) -> None:
        """Consume resources from current cell and apply metabolism.

        Harvests available sugar and spice from the current cell,
        then applies metabolism costs.
        """
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
        """Check if agent has starved.

        Returns:
            True if sugar or spice reserves are depleted.
        """
        return self.sugar <= 0 or self.spice <= 0


class SugarscapeModel(mesa.Model):  # type: ignore[misc]
    """Sugarscape model with resource gathering and trading.

    This model implements the environment logic: grid management,
    resource distribution, and regeneration mechanics. Agents use
    strategies to make decisions about movement and trading.

    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        grid: Spatial grid containing agents and resource layers.
        sugar_distribution: Maximum sugar levels for each cell.
        spice_distribution: Maximum spice levels for each cell.
    """

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
    ) -> None:
        """Initialize the Sugarscape model.

        Args:
            width: Grid width.
            height: Grid height.
            initial_population: Number of agents to create.
            endowment_min: Minimum initial resource endowment.
            endowment_max: Maximum initial resource endowment.
            metabolism_min: Minimum metabolism rate.
            metabolism_max: Maximum metabolism rate.
            vision_min: Minimum vision range.
            vision_max: Maximum vision range.
        """
        super().__init__()
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
            },
            agent_reporters={
                "Sugar": "sugar",
                "Spice": "spice",
            },
        )

    def step(self) -> None:
        """Execute one step of the simulation.

        Regenerates resources, then activates all agents in random order.
        Removes starved agents after activation.
        """
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
