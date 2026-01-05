"""Solara visualization app for Sugarscape simulation.

This module provides an interactive web-based visualization of the
Sugarscape model using Solara and Mesa's visualization components.

Run:
    PYTHONPATH=/home/makarella/kpi/thesis uv run solara run \
        src/simulations/sugarscape_app.py
"""

import os
import sys
from pathlib import Path

import instructor
import solara
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mesa.visualization import (  # noqa: E402
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from mesa.visualization.components import AgentPortrayalStyle  # noqa: E402

from src.simulations.sugarscape import SugarAgent, SugarscapeModel  # noqa: E402
from src.simulations.sugarscape_llm import (  # noqa: E402
    LLMConfig,
    SugarscapeLLMStrategy,
)
from src.simulations.sugarscape_strategy import (  # noqa: E402
    SugarscapeHeuristicStrategy,
)

load_dotenv()


def agent_portrayal(agent: SugarAgent) -> AgentPortrayalStyle:
    """Define how agents are displayed in the visualization.

    Color scheme:
    - Green shades: Healthy agents (high reserves)
    - Yellow: Warning (moderate reserves)
    - Orange: Low reserves
    - Red: Critical (very low, about to starve)
    - Blue border: LLM agent
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

    marker = "o"
    if isinstance(agent.strategy, SugarscapeLLMStrategy):
        marker = "s"

    return AgentPortrayalStyle(
        color=color,
        size=size,
        marker=marker,
    )


def create_model(
    mode: str, num_agents: int, llm_profile: str = "cached"
) -> SugarscapeModel:
    """Create a Sugarscape model with specified configuration.

    Args:
        mode: "heuristic", "llm", or "mixed"
        num_agents: Number of agents to create
        llm_profile: LLM optimization profile ("baseline", "optimized", or "cached")

    Returns:
        Configured SugarscapeModel instance
    """
    model = SugarscapeModel(width=50, height=50)

    llm_client = None
    async_llm_client = None
    llm_config = None
    if mode in ["llm", "mixed"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            llm_client = instructor.from_openai(openai_client)

            async_openai_client = AsyncOpenAI(api_key=api_key)
            async_llm_client = instructor.from_openai(async_openai_client)

            if llm_profile == "baseline":
                llm_config = LLMConfig.baseline()
            elif llm_profile == "optimized":
                llm_config = LLMConfig.optimized()
            else:
                llm_config = LLMConfig.cached()

    for i in range(num_agents):
        sugar = model.random.randint(25, 50)
        spice = model.random.randint(25, 50)
        metabolism_sugar = model.random.randint(1, 5)
        metabolism_spice = model.random.randint(1, 5)
        vision = model.random.randint(1, 5)

        x = model.random.randrange(model.width)
        y = model.random.randrange(model.height)
        cell = model.grid[(x, y)]

        if mode == "heuristic":
            strategy = SugarscapeHeuristicStrategy()
        elif mode == "llm":
            if llm_client is None:
                strategy = SugarscapeHeuristicStrategy()
            else:
                strategy = SugarscapeLLMStrategy(
                    llm_client, verbose=False, config=llm_config
                )
                strategy.async_client = async_llm_client
        else:
            if i < num_agents * 0.5:
                if llm_client is None:
                    strategy = SugarscapeHeuristicStrategy()
                else:
                    strategy = SugarscapeLLMStrategy(
                        llm_client, verbose=False, config=llm_config
                    )
                    strategy.async_client = async_llm_client
            else:
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

    return model


SpaceView = make_space_component(
    agent_portrayal,
    propertylayer_portrayal={
        "sugar": {"color": "orange", "alpha": 0.8, "vmin": 0, "vmax": 4},
        "spice": {"color": "blue", "alpha": 0.8, "vmin": 0, "vmax": 4},
    },
)
AgentCountPlot = make_plot_component("Agent Count")


@solara.component
def page():  # noqa: N802
    """Main Solara page component with configuration selector."""
    configs = [
        {
            "name": "Heuristic Agents",
            "mode": "heuristic",
            "num_agents": 80,
            "llm_profile": "cached",
            "description": "Rule-based agents using welfare maximization",
        },
        {
            "name": "LLM: Baseline (Unoptimized)",
            "mode": "llm",
            "num_agents": 5,
            "llm_profile": "baseline",
            "description": (
                "Verbose prompts, reasoning, sync calls (slowest, for comparison)"
            ),
        },
        {
            "name": "LLM: Optimized (No Cache)",
            "mode": "llm",
            "num_agents": 10,
            "llm_profile": "optimized",
            "description": "Short prompts, async calls, no cache (fast, diverse)",
        },
        {
            "name": "LLM: Cached (Fastest)",
            "mode": "llm",
            "num_agents": 10,
            "llm_profile": "cached",
            "description": "All optimizations + fuzzy caching (fastest)",
        },
        {
            "name": "Mixed Population",
            "mode": "mixed",
            "num_agents": 20,
            "llm_profile": "cached",
            "description": "50% LLM (cached) + 50% heuristic agents",
        },
    ]

    selected_config_name = solara.use_reactive(configs[0]["name"])

    selected_config = next(
        (c for c in configs if c["name"] == selected_config_name.value), configs[0]
    )

    with solara.Column(style={"width": "100%", "padding": "20px"}):
        solara.Markdown("# Sugarscape Simulation - Interactive Visualization")

        with solara.Column(
            style={
                "background-color": "#2c3e50",
                "color": "#ecf0f1",
                "padding": "15px",
                "margin-bottom": "15px",
                "border-radius": "4px",
                "border": "none",
            }
        ):
            solara.Markdown("### Legend")
            solara.Markdown(
                """
                **Agent Health (Color):**
                - 🟢 Green: Healthy (>80 reserves)
                - 🟡 Yellow: Moderate (30-80 reserves)
                - 🟠 Orange: Low (15-30 reserves)
                - 🔴 Red: Critical (<15 reserves)

                **Agent Type (Shape):**
                - ⚫ Circle: Heuristic agent (rule-based)
                - ⬛ Square: LLM agent (AI-driven)

                **Background Layers:**
                - Red overlay: Sugar distribution
                - Blue overlay: Spice distribution
                """
            )

        with solara.Column(
            style={
                "background-color": "#2c3e50",
                "color": "#ecf0f1",
                "padding": "15px",
                "margin-bottom": "15px",
                "border-radius": "4px",
                "border": "none",
            }
        ):
            solara.Select(
                label="Select Configuration",
                value=selected_config_name,
                values=[c["name"] for c in configs],
            )

            solara.Markdown(f"**{selected_config['description']}**")
            solara.Markdown(
                f"Agents: {selected_config['num_agents']} | "
                f"Mode: {selected_config['mode']} | "
                f"Profile: {selected_config['llm_profile']}"
            )

            if selected_config["mode"] in ["llm", "mixed"]:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    solara.Warning(
                        "⚠️ OPENAI_API_KEY not found. "
                        "LLM agents will fall back to heuristic mode. "
                        "Set your API key in .env file."
                    )
                else:
                    solara.Success("✓ LLM API key configured")

        model = create_model(
            mode=selected_config["mode"],
            num_agents=selected_config["num_agents"],
            llm_profile=selected_config["llm_profile"],
        )

        SolaraViz(
            model,
            components=[SpaceView, AgentCountPlot],
            agent_portrayal=agent_portrayal,
            name=selected_config["name"],
        )
