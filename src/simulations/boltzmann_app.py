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

from src.agents.strategy import DecisionStrategy  # noqa: E402
from src.simulations.boltzmann import BoltzmannAgent, BoltzmannModel  # noqa: E402
from src.simulations.boltzmann_llm import (  # noqa: E402
    BoltzmannLLMConfig,
    BoltzmannLLMStrategy,
)
from src.simulations.boltzmann_strategy import (  # noqa: E402
    BoltzmannHeuristicStrategy,
)

load_dotenv()


def agent_portrayal(agent: BoltzmannAgent) -> AgentPortrayalStyle:
    if not isinstance(agent, BoltzmannAgent):
        return AgentPortrayalStyle()

    wealth = agent.wealth

    if wealth == 0:
        color = "tab:red"
        size = 40
    elif wealth <= 2:
        color = "tab:orange"
        size = 50
    elif wealth <= 5:
        color = "gold"
        size = 60
    else:
        color = "tab:green"
        size = 70

    marker = "o"
    if isinstance(agent.strategy, BoltzmannLLMStrategy):
        marker = "s"

    return AgentPortrayalStyle(
        color=color,
        size=size,
        marker=marker,
    )


def create_model(
    mode: str, num_agents: int, llm_profile: str = "optimized"
) -> BoltzmannModel:
    model = BoltzmannModel(width=10, height=10)

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
                llm_config = BoltzmannLLMConfig.baseline()
            elif llm_profile == "optimized_with_reasoning":
                llm_config = BoltzmannLLMConfig.optimized_with_reasoning()
            else:
                llm_config = BoltzmannLLMConfig.optimized()

    cells = list(model.grid.all_cells.cells)

    for i in range(num_agents):
        cell = model.random.choice(cells)

        strategy: DecisionStrategy
        if mode == "heuristic":
            strategy = BoltzmannHeuristicStrategy(model)
        elif mode == "llm":
            if llm_client is None:
                strategy = BoltzmannHeuristicStrategy(model)
            else:
                strategy = BoltzmannLLMStrategy(
                    llm_client, verbose=False, config=llm_config
                )
                strategy.async_client = async_llm_client
        else:
            if i < num_agents * 0.5:
                if llm_client is None:
                    strategy = BoltzmannHeuristicStrategy(model)
                else:
                    strategy = BoltzmannLLMStrategy(
                        llm_client, verbose=False, config=llm_config
                    )
                    strategy.async_client = async_llm_client
            else:
                strategy = BoltzmannHeuristicStrategy(model)

        BoltzmannAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            wealth=1,
        )

    model.datacollector.collect(model)
    return model


SpaceView = make_space_component(agent_portrayal)
GiniPlot = make_plot_component("Gini Coefficient")
WealthPlot = make_plot_component("Average Wealth")


@solara.component
def page() -> None:  # noqa: N802
    configs = [
        {
            "name": "Heuristic Agents (100)",
            "mode": "heuristic",
            "num_agents": 100,
            "llm_profile": "optimized",
            "description": "Classic random movement + random wealth exchange",
        },
        {
            "name": "LLM: Baseline (Unoptimized)",
            "mode": "llm",
            "num_agents": 5,
            "llm_profile": "baseline",
            "description": "Verbose prompts, reasoning, sync calls (slowest)",
        },
        {
            "name": "LLM: Optimized",
            "mode": "llm",
            "num_agents": 10,
            "llm_profile": "optimized",
            "description": "Short prompts, async calls, no reasoning",
        },
        {
            "name": "LLM: Optimized + Reasoning",
            "mode": "llm",
            "num_agents": 10,
            "llm_profile": "optimized_with_reasoning",
            "description": "Short prompts, async calls, with reasoning",
        },
        {
            "name": "Mixed Population",
            "mode": "mixed",
            "num_agents": 20,
            "llm_profile": "optimized",
            "description": "50% LLM + 50% heuristic agents",
        },
    ]

    selected_config_name = solara.use_reactive(configs[0]["name"])

    selected_config = next(
        (c for c in configs if c["name"] == selected_config_name.value), configs[0]
    )

    with solara.Column(style={"width": "100%", "padding": "20px"}):
        solara.Markdown("# Boltzmann Wealth Model - Interactive Visualization")

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
                **Agent Wealth (Color):**
                - Red: No wealth (0 coins)
                - Orange: Low (1-2 coins)
                - Gold: Moderate (3-5 coins)
                - Green: High (6+ coins)

                **Agent Type (Shape):**
                - Circle: Heuristic agent (random exchange)
                - Square: LLM agent (AI-driven)
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
                        "OPENAI_API_KEY not found. "
                        "LLM agents will fall back to heuristic mode. "
                        "Set your API key in .env file."
                    )
                else:
                    solara.Success("LLM API key configured")

        model = create_model(
            mode=selected_config["mode"],
            num_agents=selected_config["num_agents"],
            llm_profile=selected_config["llm_profile"],
        )

        SolaraViz(
            model,
            components=[SpaceView, GiniPlot, WealthPlot],
            agent_portrayal=agent_portrayal,
            name=selected_config["name"],
        )
