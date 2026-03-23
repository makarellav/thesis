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
from src.simulations.schelling import SchellingAgent, SchellingModel  # noqa: E402
from src.simulations.schelling_llm import (  # noqa: E402
    SchellingLLMConfig,
    SchellingLLMStrategy,
)
from src.simulations.schelling_strategy import (  # noqa: E402
    SchellingHeuristicStrategy,
)

load_dotenv()


def agent_portrayal(agent: SchellingAgent) -> AgentPortrayalStyle:
    if not isinstance(agent, SchellingAgent):
        return AgentPortrayalStyle()

    color = "tab:blue" if agent.agent_type == 0 else "tab:orange"

    if isinstance(agent.strategy, SchellingLLMStrategy):
        marker = "s" if agent.happy else "D"
    else:
        marker = "o" if agent.happy else "x"

    size = 60 if agent.happy else 40

    return AgentPortrayalStyle(
        color=color,
        size=size,
        marker=marker,
    )


def create_model(
    mode: str,
    density: float = 0.8,
    homophily: float = 0.5,
    llm_profile: str = "optimized",
) -> SchellingModel:
    width = 10 if mode in ["llm", "mixed"] else 20
    height = 10 if mode in ["llm", "mixed"] else 20

    model = SchellingModel(
        width=width,
        height=height,
        density=density,
        minority_fraction=0.4,
        homophily=homophily,
    )

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
                llm_config = SchellingLLMConfig.baseline()
            elif llm_profile == "optimized_with_reasoning":
                llm_config = SchellingLLMConfig.optimized_with_reasoning()
            else:
                llm_config = SchellingLLMConfig.optimized()

    cells = list(model.grid.all_cells.cells)

    for cell in cells:
        if model.random.random() < density:
            agent_type = 1 if model.random.random() < 0.4 else 0

            strategy: DecisionStrategy
            if mode == "heuristic":
                strategy = SchellingHeuristicStrategy()
            elif mode == "llm":
                if llm_client is None:
                    strategy = SchellingHeuristicStrategy()
                else:
                    strategy = SchellingLLMStrategy(
                        llm_client, verbose=False, config=llm_config,
                        homophily=homophily,
                    )
                    strategy.async_client = async_llm_client
            else:
                if model.random.random() < 0.5:
                    if llm_client is None:
                        strategy = SchellingHeuristicStrategy()
                    else:
                        strategy = SchellingLLMStrategy(
                            llm_client, verbose=False, config=llm_config,
                            homophily=homophily,
                        )
                        strategy.async_client = async_llm_client
                else:
                    strategy = SchellingHeuristicStrategy()

            SchellingAgent(
                model=model,
                strategy=strategy,
                cell=cell,
                agent_type=agent_type,
                homophily=homophily,
            )

    model.datacollector.collect(model)
    return model


SpaceView = make_space_component(agent_portrayal)
HappyPlot = make_plot_component("Pct Happy")
SegregationPlot = make_plot_component("Segregation Index")


@solara.component
def page() -> None:  # noqa: N802
    configs = [
        {
            "name": "Heuristic Agents (20x20)",
            "mode": "heuristic",
            "llm_profile": "optimized",
            "description": "Classic Schelling: move if unhappy, stay if happy",
        },
        {
            "name": "LLM: Baseline (Unoptimized)",
            "mode": "llm",
            "llm_profile": "baseline",
            "description": "Verbose prompts, reasoning, sync calls (slowest)",
        },
        {
            "name": "LLM: Optimized",
            "mode": "llm",
            "llm_profile": "optimized",
            "description": "Short prompts, async calls, no reasoning",
        },
        {
            "name": "LLM: Optimized + Reasoning",
            "mode": "llm",
            "llm_profile": "optimized_with_reasoning",
            "description": "Short prompts, async calls, with reasoning",
        },
        {
            "name": "Mixed Population",
            "mode": "mixed",
            "llm_profile": "optimized",
            "description": "50% LLM + 50% heuristic agents",
        },
    ]

    selected_config_name = solara.use_reactive(configs[0]["name"])

    selected_config = next(
        (c for c in configs if c["name"] == selected_config_name.value), configs[0]
    )

    with solara.Column(style={"width": "100%", "padding": "20px"}):
        solara.Markdown("# Schelling Segregation Model - Interactive Visualization")

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
                - Blue: Type 0 (majority)
                - Orange: Type 1 (minority)

                **Agent State (Shape):**
                - Circle/Square: Happy (staying)
                - X/Diamond: Unhappy (will move)
                - Square/Diamond markers: LLM agents
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
            llm_profile=selected_config["llm_profile"],
        )

        SolaraViz(
            model,
            components=[SpaceView, HappyPlot, SegregationPlot],
            agent_portrayal=agent_portrayal,
            name=selected_config["name"],
        )
