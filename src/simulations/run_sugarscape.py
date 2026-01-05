"""Demo script to run the Sugarscape simulation.

This script demonstrates the Sugarscape model in action with
heuristic-based agents or LLM-based agents. It initializes the model,
creates agents with strategies, runs the simulation, and displays results.
"""

import argparse
import asyncio
import os
import time

import instructor
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.agents.strategy import DecisionStrategy
from src.simulations.sugarscape import SugarAgent, SugarscapeModel
from src.simulations.sugarscape_llm import LLMConfig, SugarscapeLLMStrategy
from src.simulations.sugarscape_strategy import SugarscapeHeuristicStrategy

load_dotenv()


async def run_agents_async(model: SugarscapeModel) -> None:
    """Run all agent steps concurrently for LLM agents.

    Args:
        model: The Sugarscape model containing agents.
    """
    agents = list(model.agents)
    model.random.shuffle(agents)

    tasks = []
    for agent in agents:
        if isinstance(agent, SugarAgent):
            if isinstance(agent.strategy, SugarscapeLLMStrategy):
                context = agent.get_context()
                task = agent.strategy.decide_async(context)
                tasks.append((agent, task))
            else:
                agent.step()

    if tasks:
        results = await asyncio.gather(*[task for _, task in tasks])
        for (agent, _), action in zip(tasks, results, strict=True):
            agent.execute_action(action)


def main() -> None:
    """Run a Sugarscape simulation demo."""
    parser = argparse.ArgumentParser(description="Run Sugarscape simulation")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "mixed"],
        default="heuristic",
        help="Agent type: heuristic (rule-based), llm (LLM-based), or mixed (both)",
    )
    parser.add_argument(
        "--llm-ratio",
        type=float,
        default=0.5,
        help="Ratio of LLM agents in mixed mode (0.0-1.0)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="Number of agents (default: 100 for heuristic, 10 for llm/mixed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM reasoning for each decision",
    )
    parser.add_argument(
        "--llm-profile",
        choices=[
            "baseline",
            "optimized",
            "cached",
            "optimized_with_reasoning",
        ],
        default="optimized",
        help=(
            "LLM optimization profile: "
            "baseline (verbose prompts + reasoning), "
            "optimized (short prompts, no reasoning), "
            "cached (optimized + fuzzy caching), "
            "optimized_with_reasoning (short prompts + reasoning)"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature for sampling (0.0-1.0, default: 0.1)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM decision caching (overrides profile setting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations (default: random)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "OpenAI model to use (default: gpt-4o-mini). "
            "Try gpt-3.5-turbo if rate limited."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Sugarscape Simulation Demo")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    llm_client = None
    async_llm_client = None
    llm_config = None
    if args.mode in ["llm", "mixed"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\n⚠ Error: OPENAI_API_KEY not found in environment variables!")
            print("Please set your API key in the .env file.")
            return

        temp = args.temperature
        if args.llm_profile == "baseline":
            llm_config = LLMConfig.baseline(temperature=temp)
        elif args.llm_profile == "optimized":
            llm_config = LLMConfig.optimized(temperature=temp)
        elif args.llm_profile == "cached":
            llm_config = LLMConfig.cached(temperature=temp)
        elif args.llm_profile == "optimized_with_reasoning":
            llm_config = LLMConfig.optimized_with_reasoning(temperature=temp)
        else:
            llm_config = LLMConfig.optimized(temperature=temp)

        if args.no_cache:
            llm_config.use_cache = False

        if args.model != "gpt-4o-mini":
            llm_config.model = args.model
            print(f"Using model: {args.model}")

        openai_client = OpenAI(api_key=api_key)
        llm_client = instructor.from_openai(openai_client)

        async_openai_client = AsyncOpenAI(api_key=api_key)
        async_llm_client = instructor.from_openai(async_openai_client)

        print("✓ LLM client initialized (gpt-4o-mini)")
        print(f"  Profile: {args.llm_profile}")
        print(f"  Prompt style: {llm_config.prompt_style}")
        print(f"  Async: {llm_config.use_async}")
        print(f"  Cache: {llm_config.use_cache}")
        print(f"  Max tokens: {llm_config.max_tokens}")
        print(f"  Temperature: {llm_config.temperature}")

    if args.num_agents is None:
        num_agents = 10 if args.mode in ["llm", "mixed"] else 100
    else:
        num_agents = args.num_agents

    if args.mode in ["llm", "mixed"] and num_agents > 20:
        print(
            f"\n⚠ Warning: Running {num_agents} LLM agents will be slow "
            f"(~{num_agents} API calls per step)"
        )
        print("Consider using --num-agents with a smaller value (e.g., 10)")

    model = SugarscapeModel(
        width=50,
        height=50,
        initial_population=num_agents,
        seed=args.seed,
    )

    print(f"\nInitializing {num_agents} agents...")

    for i in range(num_agents):
        sugar = model.random.randint(25, 50)
        spice = model.random.randint(25, 50)
        metabolism_sugar = model.random.randint(1, 5)
        metabolism_spice = model.random.randint(1, 5)
        vision = model.random.randint(1, 5)

        x = model.random.randrange(model.width)
        y = model.random.randrange(model.height)
        cell = model.grid[(x, y)]

        strategy: DecisionStrategy
        if args.mode == "heuristic":
            strategy = SugarscapeHeuristicStrategy()
        elif args.mode == "llm":
            strategy = SugarscapeLLMStrategy(
                llm_client, verbose=args.verbose, config=llm_config
            )
            strategy.async_client = async_llm_client
        else:
            if i < num_agents * args.llm_ratio:
                strategy = SugarscapeLLMStrategy(
                    llm_client, verbose=args.verbose, config=llm_config
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

    print(f"✓ Created {len(model.agents)} agents")

    num_steps = 50
    print(f"\nRunning simulation for {num_steps} steps...")
    print("-" * 60)
    print(f"{'Step':<8} {'Agents':<12} {'Avg Sugar':<12} {'Avg Spice':<12}")
    print("-" * 60)

    start_time = time.time()

    for step in range(num_steps):
        if args.mode in ["llm", "mixed"] and not args.verbose:
            print(f"Step {step}... ", end="", flush=True)

        if args.mode in ["llm", "mixed"]:
            sugar_layer = model.grid.sugar
            spice_layer = model.grid.spice
            sugar_layer.data = np.minimum(
                sugar_layer.data + 1,
                model.sugar_distribution,
            ).astype(int)
            spice_layer.data = np.minimum(
                spice_layer.data + 1,
                model.spice_distribution,
            ).astype(int)

            asyncio.run(run_agents_async(model))

            agents_to_remove = [
                agent
                for agent in model.agents
                if isinstance(agent, SugarAgent) and agent.is_starved()
            ]
            for agent in agents_to_remove:
                model.agents.remove(agent)

            model.datacollector.collect(model)
            model.steps += 1
        else:
            model.step()

        agent_count = len(model.agents)
        if agent_count > 0:
            sugar_agents = [a for a in model.agents if isinstance(a, SugarAgent)]
            avg_sugar = sum(a.sugar for a in sugar_agents) / len(sugar_agents)
            avg_spice = sum(a.spice for a in sugar_agents) / len(sugar_agents)
        else:
            avg_sugar = 0
            avg_spice = 0

        if args.mode in ["llm", "mixed"] and not args.verbose:
            print(
                f"✓ ({agent_count} agents, Avg: S={avg_sugar:.1f}, Sp={avg_spice:.1f})"
            )
        elif step % 5 == 0 or agent_count == 0:
            print(f"{step:<8} {agent_count:<12} {avg_sugar:<12.2f} {avg_spice:<12.2f}")

        if agent_count == 0:
            print("\n⚠ All agents have starved!")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 60)
    print("\nSimulation complete!")
    print(f"Final agent count: {len(model.agents)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per step: {elapsed_time / (step + 1):.3f} seconds")

    if len(model.datacollector.model_vars["Agent Count"]) > 0:
        print("\nAgent Population Over Time:")
        data = model.datacollector.model_vars["Agent Count"]
        print(f"  Initial: {data[0]}")
        print(f"  Final: {data[-1]}")
        print(f"  Survival Rate: {data[-1] / data[0] * 100:.1f}%")

    if args.mode in ["llm", "mixed"]:
        for agent in model.agents:
            if isinstance(agent, SugarAgent) and isinstance(
                agent.strategy, SugarscapeLLMStrategy
            ):
                stats = agent.strategy.get_cache_stats()
                if stats is not None:
                    print("\nLLM Cache Statistics:")
                    print(f"  Cache hits: {stats['hits']}")
                    print(f"  Cache misses: {stats['misses']}")
                    print(f"  Hit rate: {stats['hit_rate']:.1%}")
                    print(f"  Cache size: {stats['cache_size']} entries")
                    break

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
