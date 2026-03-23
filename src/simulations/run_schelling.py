import argparse
import asyncio
import os
import time

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.agents.strategy import DecisionStrategy
from src.simulations.schelling import SchellingAgent, SchellingModel
from src.simulations.schelling_llm import SchellingLLMConfig, SchellingLLMStrategy
from src.simulations.schelling_strategy import SchellingHeuristicStrategy
from src.utils.decision_logger import DecisionLogger

load_dotenv()


async def run_agents_async(model: SchellingModel) -> tuple[int, int]:
    agents = list(model.agents)
    model.random.shuffle(agents)

    skipped = 0
    tasks = []
    for agent in agents:
        if isinstance(agent, SchellingAgent):
            if isinstance(agent.strategy, SchellingLLMStrategy):
                context = agent.get_context()
                if context["happy"]:
                    agent.execute_action({"move": False})
                    skipped += 1
                else:
                    task = agent.strategy.decide_async(context)
                    tasks.append((agent, task))
            else:
                agent.step()

    if tasks:
        results = await asyncio.gather(*[task for _, task in tasks])
        for (agent, _), action in zip(tasks, results, strict=True):
            agent.execute_action(action)

    return len(tasks), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Schelling Segregation Model simulation"
    )
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
        "--density",
        type=float,
        default=0.8,
        help="Fraction of cells to populate (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--minority-fraction",
        type=float,
        default=0.4,
        help="Fraction of agents that are type 1 (0.0-1.0, default: 0.4)",
    )
    parser.add_argument(
        "--homophily",
        type=float,
        default=0.5,
        help="Minimum fraction of similar neighbors for happiness (default: 0.5)",
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
        help="Grid width (default: 20 for heuristic, 10 for llm/mixed)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Grid height (default: 20 for heuristic, 10 for llm/mixed)",
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
            "baseline (verbose prompts + reasoning), "
            "optimized (short prompts, no reasoning), "
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations (default: random)",
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
    print("Schelling Segregation Model Simulation")
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
            llm_config = SchellingLLMConfig.baseline(temperature=temp)
        elif args.llm_profile == "optimized":
            llm_config = SchellingLLMConfig.optimized(temperature=temp)
        elif args.llm_profile == "optimized_with_reasoning":
            llm_config = SchellingLLMConfig.optimized_with_reasoning(temperature=temp)
        else:
            llm_config = SchellingLLMConfig.optimized(temperature=temp)

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

    if args.width is None:
        width = 10 if args.mode in ["llm", "mixed"] else 20
    else:
        width = args.width

    if args.height is None:
        height = 10 if args.mode in ["llm", "mixed"] else 20
    else:
        height = args.height

    model = SchellingModel(
        width=width,
        height=height,
        density=args.density,
        minority_fraction=args.minority_fraction,
        homophily=args.homophily,
        seed=args.seed,
    )

    total_cells = width * height
    expected_agents = int(total_cells * args.density)
    print(f"\nInitializing agents on {width}x{height} grid...")
    print(f"  Density: {args.density} (~{expected_agents} agents)")
    print(f"  Minority fraction: {args.minority_fraction}")
    print(f"  Homophily threshold: {args.homophily}")

    cells = list(model.grid.all_cells.cells)

    logger: DecisionLogger | None = None
    if args.log_decisions:
        seed_str = f"_seed{args.seed}" if args.seed else ""
        log_path = f"schelling_decisions{seed_str}.json"
        logger = DecisionLogger(log_path)
        print(f"  Decision logging enabled → {log_path}")

    agent_count = 0
    llm_count = 0
    for cell in cells:
        if model.random.random() < args.density:
            agent_type = 1 if model.random.random() < args.minority_fraction else 0

            strategy: DecisionStrategy
            if args.mode == "heuristic":
                strategy = SchellingHeuristicStrategy(logger=logger)
            elif args.mode == "llm":
                strategy = SchellingLLMStrategy(
                    llm_client, verbose=args.verbose, config=llm_config,
                    homophily=args.homophily, logger=logger,
                )
                strategy.async_client = async_llm_client
                llm_count += 1
            else:
                if model.random.random() < args.llm_ratio:
                    strategy = SchellingLLMStrategy(
                        llm_client, verbose=args.verbose, config=llm_config,
                        homophily=args.homophily, logger=logger,
                    )
                    strategy.async_client = async_llm_client
                    llm_count += 1
                else:
                    strategy = SchellingHeuristicStrategy(logger=logger)

            SchellingAgent(
                model=model,
                strategy=strategy,
                cell=cell,
                agent_type=agent_type,
                homophily=args.homophily,
            )
            agent_count += 1

    print(f"Created {agent_count} agents")
    if llm_count > 0:
        print(f"  LLM agents: {llm_count}")
        print(f"  Heuristic agents: {agent_count - llm_count}")

    model.datacollector.collect(model)

    num_steps = args.steps
    print(f"\nRunning simulation for {num_steps} steps...")
    print("-" * 60)
    print(
        f"{'Step':<8} {'Agents':<10} {'Pct Happy':<12} "
        f"{'Segregation':<14} {'Unhappy':<10}"
    )
    print("-" * 60)

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
            pct = latest["Pct Happy"]
            print(f"done (Happy={pct:.1f}%)")
        elif step % 5 == 0 or step == num_steps - 1:
            print(
                f"{step:<8} {int(latest['Agent Count']):<10} "
                f"{latest['Pct Happy']:<12.1f} "
                f"{latest['Segregation Index']:<14.3f} "
                f"{int(latest['Num Unhappy']):<10}"
            )

        if latest["Pct Happy"] == 100.0:
            print(f"\nEquilibrium reached at step {step}!")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 60)
    print("\nSimulation complete!")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per step: {elapsed_time / max(step + 1, 1):.3f} seconds")

    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        final_metrics = df.iloc[-1]

        print("\n" + "=" * 60)
        print("FINAL METRICS:")
        print("=" * 60)
        print(f"  Agent Count: {int(final_metrics['Agent Count'])}")
        print(f"  Pct Happy: {final_metrics['Pct Happy']:.1f}%")
        print(f"  Num Happy: {int(final_metrics['Num Happy'])}")
        print(f"  Num Unhappy: {int(final_metrics['Num Unhappy'])}")
        print(f"  Segregation Index: {final_metrics['Segregation Index']:.3f}")
        print(f"  Pct Similar Neighbors: {final_metrics['Pct Similar Neighbors']:.3f}")

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
