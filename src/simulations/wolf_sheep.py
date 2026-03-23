import math
from abc import abstractmethod
from typing import Any

import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import (
    CellAgent,
    FixedAgent,
    OrthogonalVonNeumannGrid,
)

from src.agents.strategy import DecisionStrategy


class GrassPatch(FixedAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        cell: Any,
        fully_grown: bool = True,
        regrow_time: int = 30,
        countdown: int = 0,
    ) -> None:
        super().__init__(model)
        self.cell = cell
        self.fully_grown = fully_grown
        self.regrow_time = regrow_time
        self.countdown = countdown

    def step(self) -> None:
        if not self.fully_grown:
            self.countdown -= 1
            if self.countdown <= 0:
                self.fully_grown = True


class AnimalAgent(CellAgent):  # type: ignore[misc]

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
        cell: Any,
        energy: float,
        p_reproduce: float,
        energy_from_food: float,
    ) -> None:
        super().__init__(model)
        self.strategy = strategy
        self.cell = cell
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.age = 0
        self.total_moves = 0

    def step(self) -> None:
        self.age += 1
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    @abstractmethod
    def get_context(self) -> dict[str, Any]:
        ...

    def execute_action(self, action: dict[str, Any]) -> None:
        if "move" in action:
            new_cell = self.model.grid[action["move"]]
            if new_cell != self.cell:
                self.cell = new_cell
                self.total_moves += 1

        self.energy -= 1

        if action.get("eat", False):
            self._eat()

        if action.get("reproduce", False):
            self._try_reproduce()

    @abstractmethod
    def _eat(self) -> None:
        ...

    @abstractmethod
    def _try_reproduce(self) -> None:
        ...

    def is_dead(self) -> bool:
        return self.energy <= 0


class SheepAgent(AnimalAgent):

    def get_context(self) -> dict[str, Any]:
        wolves_nearby: list[dict[str, Any]] = []
        grass_neighbors: list[dict[str, Any]] = []

        for neighbor_cell in self.cell.neighborhood:
            cell_wolves = sum(
                1 for a in neighbor_cell.agents
                if isinstance(a, WolfAgent)
            )
            if cell_wolves > 0:
                wolves_nearby.append({
                    "position": neighbor_cell.coordinate,
                    "count": cell_wolves,
                })

            has_grass = any(
                isinstance(a, GrassPatch) and a.fully_grown
                for a in neighbor_cell.agents
            )
            grass_neighbors.append({
                "position": neighbor_cell.coordinate,
                "has_grass": has_grass,
            })

        current_grass = any(
            isinstance(a, GrassPatch) and a.fully_grown
            for a in self.cell.agents
        )

        sheep_in_cell = sum(
            1 for a in self.cell.agents
            if isinstance(a, SheepAgent) and a is not self
        )

        return {
            "agent_type": "sheep",
            "agent_id": self.unique_id,
            "position": self.cell.coordinate,
            "energy": self.energy,
            "age": self.age,
            "wolves_nearby": wolves_nearby,
            "wolves_nearby_count": len(wolves_nearby),
            "grass_available": current_grass,
            "grass_neighbors": grass_neighbors,
            "sheep_in_cell": sheep_in_cell,
            "cell": self.cell,
            "grid_width": self.model.width,
            "grid_height": self.model.height,
        }

    def _eat(self) -> None:
        for agent in self.cell.agents:
            if isinstance(agent, GrassPatch) and agent.fully_grown:
                agent.fully_grown = False
                agent.countdown = agent.regrow_time
                self.energy += self.energy_from_food
                break

    def _try_reproduce(self) -> None:
        if self.model.random.random() < self.p_reproduce and self.energy > 2:
            self.energy /= 2
            SheepAgent(
                model=self.model,
                strategy=self.strategy,
                cell=self.cell,
                energy=self.energy,
                p_reproduce=self.p_reproduce,
                energy_from_food=self.energy_from_food,
            )


class WolfAgent(AnimalAgent):

    def get_context(self) -> dict[str, Any]:
        sheep_nearby: list[dict[str, Any]] = []

        for neighbor_cell in self.cell.neighborhood:
            cell_sheep = sum(
                1 for a in neighbor_cell.agents
                if isinstance(a, SheepAgent)
            )
            if cell_sheep > 0:
                sheep_nearby.append({
                    "position": neighbor_cell.coordinate,
                    "count": cell_sheep,
                })

        sheep_in_cell = [
            a for a in self.cell.agents
            if isinstance(a, SheepAgent)
        ]

        wolves_in_cell = sum(
            1 for a in self.cell.agents
            if isinstance(a, WolfAgent) and a is not self
        )

        return {
            "agent_type": "wolf",
            "agent_id": self.unique_id,
            "position": self.cell.coordinate,
            "energy": self.energy,
            "age": self.age,
            "sheep_nearby": sheep_nearby,
            "sheep_nearby_count": len(sheep_nearby),
            "sheep_in_cell": len(sheep_in_cell),
            "wolves_in_cell": wolves_in_cell,
            "cell": self.cell,
            "grid_width": self.model.width,
            "grid_height": self.model.height,
        }

    def _eat(self) -> None:
        for agent in list(self.cell.agents):
            if isinstance(agent, SheepAgent):
                self.energy += self.energy_from_food
                agent.remove()
                break

    def _try_reproduce(self) -> None:
        if self.model.random.random() < self.p_reproduce and self.energy > 2:
            self.energy /= 2
            WolfAgent(
                model=self.model,
                strategy=self.strategy,
                cell=self.cell,
                energy=self.energy,
                p_reproduce=self.p_reproduce,
                energy_from_food=self.energy_from_food,
            )


class WolfSheepModel(mesa.Model):  # type: ignore[misc]

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        grass: bool = True,
        grass_regrowth_time: int = 30,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time

        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height),
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

        if self.grass:
            for cell in self.grid.all_cells.cells:
                fully_grown = self.random.random() < 0.5
                countdown = (
                    0 if fully_grown
                    else self.random.randint(0, grass_regrowth_time)
                )
                GrassPatch(
                    model=self,
                    cell=cell,
                    fully_grown=fully_grown,
                    regrow_time=grass_regrowth_time,
                    countdown=countdown,
                )

        self.datacollector = DataCollector(
            model_reporters={
                "Wolves": lambda m: _count_wolves(m),
                "Sheep": lambda m: _count_sheep(m),
                "Grass": lambda m: _count_grass(m),
                "Wolf/Sheep Ratio": lambda m: _wolf_sheep_ratio(m),
                "Total Population": lambda m: _total_population(m),
            },
        )

    def step(self) -> None:
        if self.grass:
            grass_agents = [
                a for a in self.agents if isinstance(a, GrassPatch)
            ]
            for g in grass_agents:
                g.step()

        sheep = [a for a in self.agents if isinstance(a, SheepAgent)]
        self.random.shuffle(sheep)
        for s in sheep:
            s.step()

        dead_sheep = [
            a for a in self.agents
            if isinstance(a, SheepAgent) and a.is_dead()
        ]
        for a in dead_sheep:
            a.remove()

        wolves = [a for a in self.agents if isinstance(a, WolfAgent)]
        self.random.shuffle(wolves)
        for w in wolves:
            if not w.is_dead():
                w.step()

        dead_wolves = [
            a for a in self.agents
            if isinstance(a, WolfAgent) and a.is_dead()
        ]
        for a in dead_wolves:
            a.remove()

        self.datacollector.collect(self)


def _count_wolves(model: WolfSheepModel) -> int:
    return sum(1 for a in model.agents if isinstance(a, WolfAgent))


def _count_sheep(model: WolfSheepModel) -> int:
    return sum(1 for a in model.agents if isinstance(a, SheepAgent))


def _count_grass(model: WolfSheepModel) -> int:
    return sum(
        1 for a in model.agents
        if isinstance(a, GrassPatch) and a.fully_grown
    )


def _wolf_sheep_ratio(model: WolfSheepModel) -> float:
    wolves = _count_wolves(model)
    sheep = _count_sheep(model)
    return wolves / sheep if sheep > 0 else 0.0


def _total_population(model: WolfSheepModel) -> int:
    return _count_wolves(model) + _count_sheep(model)
