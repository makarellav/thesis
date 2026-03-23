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
from src.simulations.wolf_sheep import (  # noqa: E402
    GrassPatch,
    SheepAgent,
    WolfAgent,
    WolfSheepModel,
)
from src.simulations.wolf_sheep_llm import (  # noqa: E402
    WolfSheepLLMConfig,
    WolfSheepLLMStrategy,
)
from src.simulations.wolf_sheep_strategy import (  # noqa: E402
    WolfSheepHeuristicStrategy,
)

load_dotenv()


def agent_portrayal(agent: GrassPatch | SheepAgent | WolfAgent) -> AgentPortrayalStyle:
    if isinstance(agent, GrassPatch):
        color = "tab:green" if agent.fully_grown else "saddlebrown"
        return AgentPortrayalStyle(color=color, size=20, marker="s", zorder=0)

    if isinstance(agent, WolfAgent):
        marker = "s" if isinstance(agent.strategy, WolfSheepLLMStrategy) else "o"
        return AgentPortrayalStyle(
            color="tab:red", size=60, marker=marker, zorder=2
        )

    if isinstance(agent, SheepAgent):
        marker = "s" if isinstance(agent.strategy, WolfSheepLLMStrategy) else "o"
        return AgentPortrayalStyle(
            color="white", size=50, marker=marker, zorder=1
        )

    return AgentPortrayalStyle()


def create_model(
    mode: str,
    initial_sheep: int = 100,
    initial_wolves: int = 50,
    llm_profile: str = "optimized",
) -> WolfSheepModel:
    width = 10 if mode in ["llm", "mixed"] else 20
    height = 10 if mode in ["llm", "mixed"] else 20

    model = WolfSheepModel(width=width, height=height, grass=True)

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
                llm_config = WolfSheepLLMConfig.baseline()
            elif llm_profile == "optimized_with_reasoning":
                llm_config = WolfSheepLLMConfig.optimized_with_reasoning()
            else:
                llm_config = WolfSheepLLMConfig.optimized()

    cells = list(model.grid.all_cells.cells)

    sheep_reproduce = 0.04
    wolf_reproduce = 0.05
    sheep_gain = 4.0
    wolf_gain = 20.0

    def make_strategy(agent_type: str, index: int) -> DecisionStrategy:
        if mode == "heuristic":
            return WolfSheepHeuristicStrategy(model)
        elif mode == "llm":
            if llm_client is None:
                return WolfSheepHeuristicStrategy(model)
            strategy = WolfSheepLLMStrategy(
                llm_client, verbose=False, config=llm_config
            )
            strategy.async_client = async_llm_client
            return strategy
        else:
            total = initial_sheep if agent_type == "sheep" else initial_wolves
            if index < total * 0.5:
                if llm_client is None:
                    return WolfSheepHeuristicStrategy(model)
                strategy = WolfSheepLLMStrategy(
                    llm_client, verbose=False, config=llm_config
                )
                strategy.async_client = async_llm_client
                return strategy
            return WolfSheepHeuristicStrategy(model)

    for i in range(initial_sheep):
        cell = model.random.choice(cells)
        energy = model.random.uniform(1, 2 * sheep_gain)
        strategy = make_strategy("sheep", i)
        SheepAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            energy=energy,
            p_reproduce=sheep_reproduce,
            energy_from_food=sheep_gain,
        )

    for i in range(initial_wolves):
        cell = model.random.choice(cells)
        energy = model.random.uniform(1, 2 * wolf_gain)
        strategy = make_strategy("wolf", i)
        WolfAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            energy=energy,
            p_reproduce=wolf_reproduce,
            energy_from_food=wolf_gain,
        )

    model.datacollector.collect(model)
    return model


SpaceView = make_space_component(agent_portrayal)
WolvesPlot = make_plot_component("Wolves")
SheepPlot = make_plot_component("Sheep")
GrassPlot = make_plot_component("Grass")


@solara.component
def page() -> None:  # noqa: N802
    configs = [
        {
            "name": "Heuristic Agents (100S + 50W)",
            "mode": "heuristic",
            "initial_sheep": 100,
            "initial_wolves": 50,
            "llm_profile": "optimized",
            "description": "Classic predator-prey: wolves chase, sheep flee",
        },
        {
            "name": "LLM: Baseline (20S + 10W)",
            "mode": "llm",
            "initial_sheep": 20,
            "initial_wolves": 10,
            "llm_profile": "baseline",
            "description": "Verbose prompts, reasoning, sync calls (slowest)",
        },
        {
            "name": "LLM: Optimized (20S + 10W)",
            "mode": "llm",
            "initial_sheep": 20,
            "initial_wolves": 10,
            "llm_profile": "optimized",
            "description": "Short prompts, async calls, no reasoning",
        },
        {
            "name": "LLM: Optimized + Reasoning (20S + 10W)",
            "mode": "llm",
            "initial_sheep": 20,
            "initial_wolves": 10,
            "llm_profile": "optimized_with_reasoning",
            "description": "Short prompts, async calls, with reasoning",
        },
        {
            "name": "Mixed Population (30S + 15W)",
            "mode": "mixed",
            "initial_sheep": 30,
            "initial_wolves": 15,
            "llm_profile": "optimized",
            "description": "50% LLM + 50% heuristic",
        },
    ]

    selected_config_name = solara.use_reactive(configs[0]["name"])

    selected_config = next(
        (c for c in configs if c["name"] == selected_config_name.value), configs[0]
    )

    with solara.Column(style={"width": "100%", "padding": "20px"}):
        solara.Markdown("# Wolf-Sheep Predation - Interactive Visualization")

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
                **Agent Type (Color):**
                - Red: Wolf
                - White: Sheep
                - Green/Brown: Grass (grown/regrowing)

                **Strategy (Shape):**
                - Circle: Heuristic agent
                - Square: LLM agent
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
                f"Sheep: {selected_config['initial_sheep']} | "
                f"Wolves: {selected_config['initial_wolves']} | "
                f"Mode: {selected_config['mode']}"
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
            initial_sheep=selected_config["initial_sheep"],
            initial_wolves=selected_config["initial_wolves"],
            llm_profile=selected_config["llm_profile"],
        )

        SolaraViz(
            model,
            components=[SpaceView, WolvesPlot, SheepPlot, GrassPlot],
            agent_portrayal=agent_portrayal,
            name=selected_config["name"],
        )
