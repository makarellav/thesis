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
from src.simulations.virus import (  # noqa: E402
    VirusAgent,
    VirusModel,
    VirusState,
)
from src.simulations.virus_llm import VirusLLMConfig, VirusLLMStrategy  # noqa: E402
from src.simulations.virus_strategy import VirusHeuristicStrategy  # noqa: E402

load_dotenv()


def agent_portrayal(agent: VirusAgent) -> AgentPortrayalStyle:
    if not isinstance(agent, VirusAgent):
        return AgentPortrayalStyle()

    if agent.state == VirusState.SUSCEPTIBLE:
        color = "tab:green"
    elif agent.state == VirusState.INFECTED:
        color = "tab:red"
    else:
        color = "tab:blue"

    marker = "s" if isinstance(agent.strategy, VirusLLMStrategy) else "o"
    size = 60

    return AgentPortrayalStyle(
        color=color,
        size=size,
        marker=marker,
    )


def create_model(
    mode: str,
    num_agents: int = 100,
    initial_infected: int = 5,
    llm_profile: str = "optimized",
) -> VirusModel:
    width = 10 if mode in ["llm", "mixed"] else 20
    height = 10 if mode in ["llm", "mixed"] else 20

    model = VirusModel(width=width, height=height)

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
                llm_config = VirusLLMConfig.baseline()
            elif llm_profile == "optimized_with_reasoning":
                llm_config = VirusLLMConfig.optimized_with_reasoning()
            else:
                llm_config = VirusLLMConfig.optimized()

    cells = list(model.grid.all_cells.cells)

    for i in range(num_agents):
        cell = model.random.choice(cells)
        state = (
            VirusState.INFECTED if i < initial_infected
            else VirusState.SUSCEPTIBLE
        )

        strategy: DecisionStrategy
        if mode == "heuristic":
            strategy = VirusHeuristicStrategy(model)
        elif mode == "llm":
            if llm_client is None:
                strategy = VirusHeuristicStrategy(model)
            else:
                strategy = VirusLLMStrategy(
                    llm_client, verbose=False, config=llm_config
                )
                strategy.async_client = async_llm_client
        else:
            if i < num_agents * 0.5:
                if llm_client is None:
                    strategy = VirusHeuristicStrategy(model)
                else:
                    strategy = VirusLLMStrategy(
                        llm_client, verbose=False, config=llm_config
                    )
                    strategy.async_client = async_llm_client
            else:
                strategy = VirusHeuristicStrategy(model)

        VirusAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            state=state,
        )

    model.datacollector.collect(model)
    return model


SpaceView = make_space_component(agent_portrayal)
InfectedPlot = make_plot_component("Infected")
SusceptiblePlot = make_plot_component("Susceptible")
ResistantPlot = make_plot_component("Resistant")


@solara.component
def page() -> None:  # noqa: N802
    configs = [
        {
            "name": "Heuristic Agents (100)",
            "mode": "heuristic",
            "num_agents": 100,
            "initial_infected": 5,
            "llm_profile": "optimized",
            "description": "Classic SIR: random movement + random spread",
        },
        {
            "name": "LLM: Baseline (Unoptimized)",
            "mode": "llm",
            "num_agents": 10,
            "initial_infected": 2,
            "llm_profile": "baseline",
            "description": "Verbose prompts, reasoning, sync calls (slowest)",
        },
        {
            "name": "LLM: Optimized",
            "mode": "llm",
            "num_agents": 10,
            "initial_infected": 2,
            "llm_profile": "optimized",
            "description": "Short prompts, async calls, no reasoning",
        },
        {
            "name": "LLM: Optimized + Reasoning",
            "mode": "llm",
            "num_agents": 10,
            "initial_infected": 2,
            "llm_profile": "optimized_with_reasoning",
            "description": "Short prompts, async calls, with reasoning",
        },
        {
            "name": "Mixed Population",
            "mode": "mixed",
            "num_agents": 20,
            "initial_infected": 3,
            "llm_profile": "optimized",
            "description": "50% LLM + 50% heuristic agents",
        },
    ]

    selected_config_name = solara.use_reactive(configs[0]["name"])

    selected_config = next(
        (c for c in configs if c["name"] == selected_config_name.value), configs[0]
    )

    with solara.Column(style={"width": "100%", "padding": "20px"}):
        solara.Markdown("# Virus Spread (SIR) Model - Interactive Visualization")

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
                **Agent State (Color):**
                - Green: Susceptible
                - Red: Infected
                - Blue: Resistant

                **Agent Type (Shape):**
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
            initial_infected=selected_config["initial_infected"],
            llm_profile=selected_config["llm_profile"],
        )

        SolaraViz(
            model,
            components=[SpaceView, InfectedPlot, SusceptiblePlot, ResistantPlot],
            agent_portrayal=agent_portrayal,
            name=selected_config["name"],
        )
