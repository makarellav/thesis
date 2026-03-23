from typing import Any

import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

from src.agents.strategy import DecisionStrategy
from src.utils.metrics import calculate_gini


class BoltzmannAgent(CellAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
        cell: Any,
        wealth: int = 1,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.wealth = wealth
        self.age = 0
        self.total_trades = 0
        self.suboptimal_trades = 0

    def step(self) -> None:
        self.age += 1
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        current_pos = self.cell.coordinate

        neighbors = []
        for neighbor_cell in self.cell.neighborhood:
            cell_agents = [
                a for a in neighbor_cell.agents
                if isinstance(a, BoltzmannAgent) and a is not self
            ]
            neighbors.append({
                "position": neighbor_cell.coordinate,
                "agent_count": len(cell_agents),
                "total_wealth": sum(a.wealth for a in cell_agents),
            })

        cellmates = [
            a for a in self.cell.agents
            if isinstance(a, BoltzmannAgent) and a is not self
        ]

        return {
            "agent_id": self.unique_id,
            "position": current_pos,
            "wealth": self.wealth,
            "neighbors": neighbors,
            "cellmates": [{"id": a.unique_id, "wealth": a.wealth} for a in cellmates],
            "cell": self.cell,
            "grid_width": self.model.width,
            "grid_height": self.model.height,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        if "move" in action:
            new_pos = action["move"]
            new_cell = self.model.grid[new_pos]
            self.cell = new_cell

        if action.get("give", False) and self.wealth > 0:
            cellmates = [
                a for a in self.cell.agents
                if isinstance(a, BoltzmannAgent) and a is not self
            ]
            if cellmates:
                recipient = self.model.random.choice(cellmates)
                recipient.wealth += 1
                self.wealth -= 1
                self.total_trades += 1

                if not self._is_optimal_trade():
                    self.suboptimal_trades += 1

    def _is_optimal_trade(self) -> bool:
        cellmates = [
            a for a in self.cell.agents
            if isinstance(a, BoltzmannAgent) and a is not self
        ]
        if not cellmates:
            return True

        avg_cellmate_wealth = sum(a.wealth for a in cellmates) / len(cellmates)
        return self.wealth >= avg_cellmate_wealth


class BoltzmannModel(mesa.Model):  # type: ignore[misc]

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height

        self.grid = OrthogonalMooreGrid(
            (self.width, self.height),
            torus=True,
            random=self.random,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Agent Count": lambda m: len(m.agents),
                "Gini Coefficient": lambda m: calculate_gini(
                    [
                        float(a.wealth)
                        for a in m.agents
                        if isinstance(a, BoltzmannAgent)
                    ]
                ),
                "Average Wealth": lambda m: (
                    sum(a.wealth for a in m.agents if isinstance(a, BoltzmannAgent))
                    / len(m.agents)
                    if len(m.agents) > 0
                    else 0
                ),
                "Wealth Std Dev": lambda m: _wealth_std(m),
                "Suboptimal Trade Rate": lambda m: (
                    sum(
                        a.suboptimal_trades
                        for a in m.agents
                        if isinstance(a, BoltzmannAgent)
                    )
                    / sum(
                        a.total_trades
                        for a in m.agents
                        if isinstance(a, BoltzmannAgent)
                    )
                    if sum(
                        a.total_trades
                        for a in m.agents
                        if isinstance(a, BoltzmannAgent)
                    )
                    > 0
                    else 0
                ),
            },
            agent_reporters={
                "Wealth": "wealth",
                "Age": "age",
                "Total Trades": "total_trades",
                "Suboptimal Trades": "suboptimal_trades",
            },
        )

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def _wealth_std(model: BoltzmannModel) -> float:
    agents = [a for a in model.agents if isinstance(a, BoltzmannAgent)]
    if not agents:
        return 0.0
    wealths = [float(a.wealth) for a in agents]
    mean = sum(wealths) / len(wealths)
    variance = sum((w - mean) ** 2 for w in wealths) / len(wealths)
    return float(variance**0.5)
