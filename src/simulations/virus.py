from enum import Enum
from typing import Any

import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

from src.agents.strategy import DecisionStrategy


class VirusState(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


class VirusAgent(CellAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
        cell: Any,
        state: VirusState = VirusState.SUSCEPTIBLE,
        virus_spread_chance: float = 0.4,
        virus_check_frequency: float = 0.4,
        recovery_chance: float = 0.3,
        gain_resistance_chance: float = 0.5,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.state = state
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance
        self.times_infected = 1 if state == VirusState.INFECTED else 0
        self.infection_duration = 0
        self.times_spread = 0

    def step(self) -> None:
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        cellmates = [
            a for a in self.cell.agents
            if isinstance(a, VirusAgent) and a is not self
        ]

        neighbors = []
        infected_neighbors = 0
        susceptible_neighbors = 0
        resistant_neighbors = 0
        for neighbor_cell in self.cell.neighborhood:
            cell_agents = [
                a for a in neighbor_cell.agents
                if isinstance(a, VirusAgent)
            ]
            cell_infected = sum(
                1 for a in cell_agents if a.state == VirusState.INFECTED
            )
            cell_susceptible = sum(
                1 for a in cell_agents if a.state == VirusState.SUSCEPTIBLE
            )
            cell_resistant = sum(
                1 for a in cell_agents if a.state == VirusState.RESISTANT
            )
            infected_neighbors += cell_infected
            susceptible_neighbors += cell_susceptible
            resistant_neighbors += cell_resistant
            neighbors.append({
                "position": neighbor_cell.coordinate,
                "infected": cell_infected,
                "susceptible": cell_susceptible,
                "resistant": cell_resistant,
                "total": len(cell_agents),
            })

        total_neighbors = (
            infected_neighbors + susceptible_neighbors + resistant_neighbors
        )
        infection_risk = (
            infected_neighbors / total_neighbors if total_neighbors > 0 else 0.0
        )

        cellmate_infected = sum(
            1 for c in cellmates if c.state == VirusState.INFECTED
        )

        return {
            "agent_id": self.unique_id,
            "position": self.cell.coordinate,
            "state": self.state.name,
            "infected_neighbors": infected_neighbors,
            "susceptible_neighbors": susceptible_neighbors,
            "resistant_neighbors": resistant_neighbors,
            "total_neighbors": total_neighbors,
            "infection_risk": infection_risk,
            "cellmate_infected": cellmate_infected,
            "cellmate_count": len(cellmates),
            "infection_duration": self.infection_duration,
            "cell": self.cell,
            "grid_width": self.model.width,
            "grid_height": self.model.height,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        if "move" in action:
            new_pos = action["move"]
            new_cell = self.model.grid[new_pos]
            self.cell = new_cell

        if self.state == VirusState.INFECTED:
            self.infection_duration += 1

            if action.get("interact", True):
                self._try_infect_cellmates()

            self._check_recovery()

    def _try_infect_cellmates(self) -> None:
        cellmates = [
            a for a in self.cell.agents
            if isinstance(a, VirusAgent) and a is not self
            and a.state == VirusState.SUSCEPTIBLE
        ]
        for agent in cellmates:
            if self.model.random.random() < self.virus_spread_chance:
                agent.state = VirusState.INFECTED
                agent.times_infected += 1
                agent.infection_duration = 0
                self.times_spread += 1

    def _check_recovery(self) -> None:
        if (
            self.model.random.random() < self.virus_check_frequency
            and self.model.random.random() < self.recovery_chance
        ):
            if self.model.random.random() < self.gain_resistance_chance:
                self.state = VirusState.RESISTANT
            else:
                self.state = VirusState.SUSCEPTIBLE


class VirusModel(mesa.Model):  # type: ignore[misc]

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        virus_spread_chance: float = 0.4,
        virus_check_frequency: float = 0.4,
        recovery_chance: float = 0.3,
        gain_resistance_chance: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            torus=True,
            random=self.random,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Infected": lambda m: _count_state(m, VirusState.INFECTED),
                "Susceptible": lambda m: _count_state(
                    m, VirusState.SUSCEPTIBLE
                ),
                "Resistant": lambda m: _count_state(m, VirusState.RESISTANT),
                "Pct Infected": lambda m: _pct_infected(m),
                "Attack Rate": lambda m: _attack_rate(m),
                "Agent Count": lambda m: len(m.agents),
            },
            agent_reporters={
                "State": lambda a: a.state.name
                if isinstance(a, VirusAgent) else "N/A",
                "Times Infected": lambda a: a.times_infected
                if isinstance(a, VirusAgent) else 0,
            },
        )

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def _count_state(model: VirusModel, state: VirusState) -> int:
    return sum(
        1 for a in model.agents
        if isinstance(a, VirusAgent) and a.state == state
    )


def _pct_infected(model: VirusModel) -> float:
    agents = [a for a in model.agents if isinstance(a, VirusAgent)]
    if not agents:
        return 0.0
    infected = sum(1 for a in agents if a.state == VirusState.INFECTED)
    return float(infected / len(agents) * 100)


def _attack_rate(model: VirusModel) -> float:
    agents = [a for a in model.agents if isinstance(a, VirusAgent)]
    if not agents:
        return 0.0
    ever_infected = sum(1 for a in agents if a.times_infected > 0)
    return float(ever_infected / len(agents) * 100)
