import argparse
import asyncio
import os
import time

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.agents.strategy import DecisionStrategy
from src.simulations.wolf_sheep import (
    GrassPatch,
    SheepAgent,
    WolfAgent,
    WolfSheepModel,
)
from src.simulations.wolf_sheep_llm import WolfSheepLLMConfig, WolfSheepLLMStrategy
from src.simulations.wolf_sheep_strategy import WolfSheepHeuristicStrategy
from src.utils.decision_logger import DecisionLogger

load_dotenv()


async def run_agents_async(model: WolfSheepModel) -> None:
    if model.grass:
        grass_agents = [
            a for a in model.agents if isinstance(a, GrassPatch)
        ]
        for g in grass_agents:
            g.step()

    sheep = [a for a in model.agents if isinstance(a, SheepAgent)]
    model.random.shuffle(sheep)

    sheep_tasks = []
    for s in sheep:
        if isinstance(s.strategy, WolfSheepLLMStrategy):
            context = s.get_context()
            task = s.strategy.decide_async(context)
            sheep_tasks.append((s, task))
        else:
            s.step()

    if sheep_tasks:
        results = await asyncio.gather(*[task for _, task in sheep_tasks])
        for (agent, _), action in zip(sheep_tasks, results, strict=True):
            agent.age += 1
            agent.execute_action(action)

    dead_sheep = [
        a for a in model.agents
        if isinstance(a, SheepAgent) and a.is_dead()
    ]
    for a in dead_sheep:
        a.remove()

    wolves = [a for a in model.agents if isinstance(a, WolfAgent)]
    model.random.shuffle(wolves)

    wolf_tasks = []
    for w in wolves:
        if not w.is_dead():
            if isinstance(w.strategy, WolfSheepLLMStrategy):
                context = w.get_context()
                task = w.strategy.decide_async(context)
                wolf_tasks.append((w, task))
            else:
                w.step()

    if wolf_tasks:
        results = await asyncio.gather(*[task for _, task in wolf_tasks])
        for (agent, _), action in zip(wolf_tasks, results, strict=True):
            agent.age += 1
            agent.execute_action(action)

    dead_wolves = [
        a for a in model.agents
        if isinstance(a, WolfAgent) and a.is_dead()
    ]
    for a in dead_wolves:
        a.remove()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wolf-Sheep Predation Model simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "mixed"],
        default="heuristic",
        help="Agent type: heuristic, llm, or mixed",
    )
    parser.add_argument(
        "--llm-ratio",
        type=float,
        default=0.5,
        help="Ratio of LLM agents in mixed mode (0.0-1.0)",
    )
    parser.add_argument(
        "--initial-sheep",
        type=int,
        default=None,
        help="Number of sheep (default: 100 heuristic, 20 llm)",
    )
    parser.add_argument(
        "--initial-wolves",
        type=int,
        default=None,
        help="Number of wolves (default: 50 heuristic, 10 llm)",
    )
    parser.add_argument(
        "--sheep-reproduce",
        type=float,
        default=0.04,
        help="Sheep reproduction probability (default: 0.04)",
    )
    parser.add_argument(
        "--wolf-reproduce",
        type=float,
        default=0.05,
        help="Wolf reproduction probability (default: 0.05)",
    )
    parser.add_argument(
        "--wolf-gain-from-food",
        type=float,
        default=20.0,
        help="Energy wolf gains from eating sheep (default: 20)",
    )
    parser.add_argument(
        "--sheep-gain-from-food",
        type=float,
        default=4.0,
        help="Energy sheep gains from eating grass (default: 4)",
    )
    parser.add_argument(
        "--grass-regrowth-time",
        type=int,
        default=30,
        help="Steps for grass to regrow (default: 30)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of simulation steps (default: 50)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Grid width (default: 20 heuristic, 10 llm)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Grid height (default: 20 heuristic, 10 llm)",
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
            "optimized_with_reasoning",
        ],
        default="optimized",
        help=(
            "LLM optimization profile: "
            "baseline, optimized, optimized_with_reasoning"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (0.0-1.0, default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--log-decisions",
        action="store_true",
        help="Log all agent decisions to JSON file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Save per-step metrics DataFrame to CSV file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Wolf-Sheep Predation Model Simulation")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    llm_client = None
    async_llm_client = None
    llm_config = None
    if args.mode in ["llm", "mixed"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\nError: OPENAI_API_KEY not found in environment variables!")
            print("Please set your API key in the .env file.")
            return

        temp = args.temperature
        if args.llm_profile == "baseline":
            llm_config = WolfSheepLLMConfig.baseline(temperature=temp)
        elif args.llm_profile == "optimized":
            llm_config = WolfSheepLLMConfig.optimized(temperature=temp)
        elif args.llm_profile == "optimized_with_reasoning":
            llm_config = WolfSheepLLMConfig.optimized_with_reasoning(
                temperature=temp
            )
        else:
            llm_config = WolfSheepLLMConfig.optimized(temperature=temp)

        if args.model != "gpt-4o-mini":
            llm_config.model = args.model
            print(f"Using model: {args.model}")

        openai_client = OpenAI(api_key=api_key)
        llm_client = instructor.from_openai(openai_client)

        async_openai_client = AsyncOpenAI(api_key=api_key)
        async_llm_client = instructor.from_openai(async_openai_client)

        print("LLM client initialized")
        print(f"  Profile: {args.llm_profile}")
        print(f"  Prompt style: {llm_config.prompt_style}")
        print(f"  Async: {llm_config.use_async}")
        print(f"  Max tokens: {llm_config.max_tokens}")
        print(f"  Temperature: {llm_config.temperature}")

    width = (
        args.width if args.width is not None
        else (10 if args.mode in ["llm", "mixed"] else 20)
    )
    height = (
        args.height if args.height is not None
        else (10 if args.mode in ["llm", "mixed"] else 20)
    )
    initial_sheep = (
        args.initial_sheep if args.initial_sheep is not None
        else (20 if args.mode in ["llm", "mixed"] else 100)
    )
    initial_wolves = (
        args.initial_wolves if args.initial_wolves is not None
        else (10 if args.mode in ["llm", "mixed"] else 50)
    )

    model = WolfSheepModel(
        width=width,
        height=height,
        grass=True,
        grass_regrowth_time=args.grass_regrowth_time,
        seed=args.seed,
    )

    print(f"\nInitializing on {width}x{height} grid...")
    print(f"  Sheep: {initial_sheep} (reproduce: {args.sheep_reproduce})")
    print(f"  Wolves: {initial_wolves} (reproduce: {args.wolf_reproduce})")
    print(f"  Sheep gain from food: {args.sheep_gain_from_food}")
    print(f"  Wolf gain from food: {args.wolf_gain_from_food}")
    print(f"  Grass regrowth time: {args.grass_regrowth_time}")

    cells = list(model.grid.all_cells.cells)

    logger: DecisionLogger | None = None
    if args.log_decisions:
        seed_str = f"_seed{args.seed}" if args.seed else ""
        log_path = f"wolfsheep_decisions{seed_str}.json"
        logger = DecisionLogger(log_path)
        print(f"  Decision logging enabled → {log_path}")

    def make_strategy(agent_type: str, index: int) -> DecisionStrategy:
        if args.mode == "heuristic":
            return WolfSheepHeuristicStrategy(model, logger=logger)
        elif args.mode == "llm":
            strategy = WolfSheepLLMStrategy(
                llm_client, verbose=args.verbose, config=llm_config,
                logger=logger,
            )
            strategy.async_client = async_llm_client
            return strategy
        else:
            total = initial_sheep if agent_type == "sheep" else initial_wolves
            if index < total * args.llm_ratio:
                strategy = WolfSheepLLMStrategy(
                    llm_client, verbose=args.verbose, config=llm_config,
                    logger=logger,
                )
                strategy.async_client = async_llm_client
                return strategy
            return WolfSheepHeuristicStrategy(model, logger=logger)

    for i in range(initial_sheep):
        cell = model.random.choice(cells)
        energy = model.random.uniform(1, 2 * args.sheep_gain_from_food)
        strategy = make_strategy("sheep", i)
        SheepAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            energy=energy,
            p_reproduce=args.sheep_reproduce,
            energy_from_food=args.sheep_gain_from_food,
        )

    for i in range(initial_wolves):
        cell = model.random.choice(cells)
        energy = model.random.uniform(1, 2 * args.wolf_gain_from_food)
        strategy = make_strategy("wolf", i)
        WolfAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            energy=energy,
            p_reproduce=args.wolf_reproduce,
            energy_from_food=args.wolf_gain_from_food,
        )

    sheep_count = sum(
        1 for a in model.agents if isinstance(a, SheepAgent)
    )
    wolf_count = sum(
        1 for a in model.agents if isinstance(a, WolfAgent)
    )
    print(f"Created {sheep_count} sheep and {wolf_count} wolves")

    model.datacollector.collect(model)

    num_steps = args.steps
    print(f"\nRunning simulation for {num_steps} steps...")
    print("-" * 70)
    print(
        f"{'Step':<8} {'Wolves':<10} {'Sheep':<10} "
        f"{'Grass':<10} {'W/S Ratio':<12} {'Total Pop':<10}"
    )
    print("-" * 70)

    start_time = time.time()

    for step in range(num_steps):
        if logger is not None:
            logger.step_counter = step

        if args.mode in ["llm", "mixed"] and not args.verbose:
            print(f"Step {step}... ", end="", flush=True)

        if args.mode in ["llm", "mixed"]:
            asyncio.run(run_agents_async(model))
            model.datacollector.collect(model)
            model.steps += 1
        else:
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        latest = df.iloc[-1]

        if args.mode in ["llm", "mixed"] and not args.verbose:
            print(
                f"done (W={int(latest['Wolves'])}, "
                f"S={int(latest['Sheep'])})"
            )
        elif step % 5 == 0 or step == num_steps - 1:
            print(
                f"{step:<8} {int(latest['Wolves']):<10} "
                f"{int(latest['Sheep']):<10} "
                f"{int(latest['Grass']):<10} "
                f"{latest['Wolf/Sheep Ratio']:<12.2f} "
                f"{int(latest['Total Population']):<10}"
            )

        if int(latest['Wolves']) == 0 and int(latest['Sheep']) == 0:
            print("\n*** Both populations extinct! ***")
            break
        if int(latest['Wolves']) == 0:
            print(f"\n*** Wolves extinct at step {step}! ***")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_steps = step + 1

    print("-" * 70)
    print("\nSimulation complete!")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per step: {elapsed_time / actual_steps:.3f} seconds")

    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        final_metrics = df.iloc[-1]
        peak_wolves = int(df["Wolves"].max())
        peak_sheep = int(df["Sheep"].max())

        print("\n" + "=" * 60)
        print("FINAL METRICS:")
        print("=" * 60)
        print(f"  Wolves: {int(final_metrics['Wolves'])}")
        print(f"  Sheep: {int(final_metrics['Sheep'])}")
        print(f"  Grass: {int(final_metrics['Grass'])}")
        print(f"  Wolf/Sheep Ratio: {final_metrics['Wolf/Sheep Ratio']:.2f}")
        print(f"  Total Population: {int(final_metrics['Total Population'])}")
        print(f"  Peak Wolves: {peak_wolves}")
        print(f"  Peak Sheep: {peak_sheep}")
        print(f"  Steps Completed: {actual_steps}")

    if args.output_csv and not df.empty:
        df.to_csv(args.output_csv, index_label="Step")
        print(f"  CSV saved: {args.output_csv}")

    if logger is not None:
        logger.save()
        summary = logger.get_summary()
        print("\nDECISION LOG SUMMARY:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Heuristic: {summary['heuristic_decisions']}")
        print(f"  LLM: {summary['llm_decisions']}")
        print(f"  Errors: {summary['errors']}")
        print(f"  Saved to: {logger.output_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
