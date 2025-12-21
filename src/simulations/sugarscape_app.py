"""Solara visualization app for Sugarscape simulation.

This module provides an interactive web-based visualization of the
Sugarscape model using Solara and Mesa's visualization components.

Run: uv run solara run src/simulations/sugarscape_app.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mesa.visualization import (  # noqa: E402
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from mesa.visualization.components import AgentPortrayalStyle  # noqa: E402

from src.simulations.sugarscape import SugarAgent, SugarscapeModel  # noqa: E402
from src.simulations.sugarscape_strategy import (  # noqa: E402
    SugarscapeHeuristicStrategy,
)


def agent_portrayal(agent: SugarAgent) -> AgentPortrayalStyle:
    """Define how agents are displayed in the visualization.

    Color scheme:
    - Green: Healthy agents (high reserves)
    - Yellow: Warning (moderate reserves)
    - Orange: Low reserves
    - Red: Critical (very low, about to starve)
    - Hidden: Dead/starved agents (not displayed)
    """
    if not isinstance(agent, SugarAgent):
        return AgentPortrayalStyle()

    if agent.is_starved():
        return AgentPortrayalStyle(
            color="white",
            size=0,
            alpha=0.0,
        )

    total_reserves = agent.sugar + agent.spice

    if total_reserves > 80:
        color = "tab:green"
        size = 65
    elif total_reserves > 50:
        color = "limegreen"
        size = 60
    elif total_reserves > 30:
        color = "gold"
        size = 55
    elif total_reserves > 15:
        color = "tab:orange"
        size = 50
    else:
        color = "tab:red"
        size = 45

    return AgentPortrayalStyle(
        color=color,
        size=size,
    )


model = SugarscapeModel(width=50, height=50)

for _i in range(80):
    sugar = model.random.randint(25, 50)
    spice = model.random.randint(25, 50)
    metabolism_sugar = model.random.randint(1, 5)
    metabolism_spice = model.random.randint(1, 5)
    vision = model.random.randint(1, 5)

    x = model.random.randrange(model.width)
    y = model.random.randrange(model.height)
    cell = model.grid[(x, y)]

    strategy = SugarscapeHeuristicStrategy()
    SugarAgent(
        model=model,
        strategy=strategy,
        cell=cell,
        sugar=sugar,
        spice=spice,
        metabolism_sugar=metabolism_sugar,
        metabolism_spice=metabolism_spice,
        vision=vision,
    )

SpaceView = make_space_component(
    agent_portrayal,
    propertylayer_portrayal={
        "sugar": {"colormap": "YlOrRd", "alpha": 0.6},
        "spice": {"colormap": "Blues", "alpha": 0.6},
    },
)
AgentCountPlot = make_plot_component("Agent Count")

page = SolaraViz(
    model,
    components=[SpaceView, AgentCountPlot],
    agent_portrayal=agent_portrayal,
    name="Sugarscape - Heuristic Agents",
)
