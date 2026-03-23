from typing import Any

import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

from src.agents.strategy import DecisionStrategy


class SchellingAgent(CellAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
        cell: Any,
        agent_type: int,
        homophily: float = 0.5,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.agent_type = agent_type
        self.homophily = homophily
        self.happy = False
        self.times_moved = 0
        self.times_stayed = 0

    def step(self) -> None:
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        neighbors = list(self.cell.neighborhood.agents)

        similar_count = sum(
            1 for n in neighbors
            if isinstance(n, SchellingAgent) and n.agent_type == self.agent_type
        )
        different_count = sum(
            1 for n in neighbors
            if isinstance(n, SchellingAgent) and n.agent_type != self.agent_type
        )
        total_neighbors = similar_count + different_count

        similar_fraction = (
            similar_count / total_neighbors if total_neighbors > 0 else 0.0
        )

        return {
            "agent_id": self.unique_id,
            "position": self.cell.coordinate,
            "agent_type": self.agent_type,
            "homophily": self.homophily,
            "similar_count": similar_count,
            "different_count": different_count,
            "total_neighbors": total_neighbors,
            "similar_fraction": similar_fraction,
            "happy": similar_fraction >= self.homophily or total_neighbors == 0,
            "grid_width": self.model.width,
            "grid_height": self.model.height,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        if action.get("move", False):
            self.cell = self.model.grid.select_random_empty_cell()
            self.times_moved += 1
        else:
            self.times_stayed += 1

        self._update_happiness()

    def _update_happiness(self) -> None:
        neighbors = list(self.cell.neighborhood.agents)
        schelling_neighbors = [
            n for n in neighbors if isinstance(n, SchellingAgent)
        ]
        if not schelling_neighbors:
            self.happy = True
            return

        similar = sum(
            1 for n in schelling_neighbors
            if n.agent_type == self.agent_type
        )
        self.happy = similar / len(schelling_neighbors) >= self.homophily


class SchellingModel(mesa.Model):  # type: ignore[misc]

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        density: float = 0.8,
        minority_fraction: float = 0.4,
        homophily: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.density = density
        self.minority_fraction = minority_fraction
        self.homophily = homophily

        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            torus=True,
            random=self.random,
            capacity=1,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Pct Happy": lambda m: _pct_happy(m),
                "Pct Similar Neighbors": lambda m: _avg_similar_fraction(m),
                "Num Happy": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, SchellingAgent) and a.happy
                ),
                "Num Unhappy": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, SchellingAgent) and not a.happy
                ),
                "Segregation Index": lambda m: _avg_similar_fraction(m),
                "Agent Count": lambda m: len(m.agents),
            },
            agent_reporters={
                "Agent Type": "agent_type",
                "Happy": "happy",
                "Times Moved": "times_moved",
            },
        )

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def _pct_happy(model: SchellingModel) -> float:
    agents = [a for a in model.agents if isinstance(a, SchellingAgent)]
    if not agents:
        return 0.0
    happy_count = sum(1 for a in agents if a.happy)
    return float(happy_count / len(agents) * 100)


def _avg_similar_fraction(model: SchellingModel) -> float:
    agents = [a for a in model.agents if isinstance(a, SchellingAgent)]
    if not agents:
        return 0.0

    total_fraction = 0.0
    count = 0
    for agent in agents:
        neighbors = list(agent.cell.neighborhood.agents)
        schelling_neighbors = [
            n for n in neighbors if isinstance(n, SchellingAgent)
        ]
        if schelling_neighbors:
            similar = sum(
                1 for n in schelling_neighbors
                if n.agent_type == agent.agent_type
            )
            total_fraction += similar / len(schelling_neighbors)
            count += 1

    return float(total_fraction / count) if count > 0 else 0.0
