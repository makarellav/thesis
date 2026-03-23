import argparse
import asyncio
import os
import time

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from src.agents.strategy import DecisionStrategy
from src.simulations.virus import VirusAgent, VirusModel, VirusState
from src.simulations.virus_llm import VirusLLMConfig, VirusLLMStrategy
from src.simulations.virus_strategy import VirusHeuristicStrategy
from src.utils.decision_logger import DecisionLogger

load_dotenv()


async def run_agents_async(model: VirusModel) -> None:
    agents = list(model.agents)
    model.random.shuffle(agents)

    tasks = []
    for agent in agents:
        if isinstance(agent, VirusAgent):
            if isinstance(agent.strategy, VirusLLMStrategy):
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
    parser = argparse.ArgumentParser(
        description="Run Virus Spread (SIR) Model simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic", "llm", "mixed"],
        default="heuristic",
        help="Agent type: heuristic (rule-based), llm (LLM-based), or mixed",
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
        "--initial-infected",
        type=int,
        default=None,
        help="Number of initially infected agents (default: 5 heuristic, 2 llm)",
    )
    parser.add_argument(
        "--virus-spread-chance",
        type=float,
        default=0.4,
        help="Probability of spreading to a neighbor (default: 0.4)",
    )
    parser.add_argument(
        "--virus-check-frequency",
        type=float,
        default=0.4,
        help="Probability of checking infection each step (default: 0.4)",
    )
    parser.add_argument(
        "--recovery-chance",
        type=float,
        default=0.3,
        help="Probability of recovering when checked (default: 0.3)",
    )
    parser.add_argument(
        "--gain-resistance-chance",
        type=float,
        default=0.5,
        help="Probability of resistance after recovery (default: 0.5)",
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
    print("Virus Spread (SIR) Model Simulation")
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
            llm_config = VirusLLMConfig.baseline(temperature=temp)
        elif args.llm_profile == "optimized":
            llm_config = VirusLLMConfig.optimized(temperature=temp)
        elif args.llm_profile == "optimized_with_reasoning":
            llm_config = VirusLLMConfig.optimized_with_reasoning(temperature=temp)
        else:
            llm_config = VirusLLMConfig.optimized(temperature=temp)

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

    if args.num_agents is None:
        num_agents = 10 if args.mode in ["llm", "mixed"] else 100
    else:
        num_agents = args.num_agents

    if args.initial_infected is None:
        initial_infected = 2 if args.mode in ["llm", "mixed"] else 5
    else:
        initial_infected = args.initial_infected

    model = VirusModel(
        width=width,
        height=height,
        virus_spread_chance=args.virus_spread_chance,
        virus_check_frequency=args.virus_check_frequency,
        recovery_chance=args.recovery_chance,
        gain_resistance_chance=args.gain_resistance_chance,
        seed=args.seed,
    )

    print(f"\nInitializing {num_agents} agents on {width}x{height} grid...")
    print(f"  Initially infected: {initial_infected}")
    print(f"  Spread chance: {args.virus_spread_chance}")
    print(f"  Recovery chance: {args.recovery_chance}")
    print(f"  Resistance chance: {args.gain_resistance_chance}")

    cells = list(model.grid.all_cells.cells)

    logger: DecisionLogger | None = None
    if args.log_decisions:
        seed_str = f"_seed{args.seed}" if args.seed else ""
        log_path = f"virus_decisions{seed_str}.json"
        logger = DecisionLogger(log_path)
        print(f"  Decision logging enabled → {log_path}")

    for i in range(num_agents):
        cell = model.random.choice(cells)
        state = (
            VirusState.INFECTED if i < initial_infected
            else VirusState.SUSCEPTIBLE
        )

        strategy: DecisionStrategy
        if args.mode == "heuristic":
            strategy = VirusHeuristicStrategy(model, logger=logger)
        elif args.mode == "llm":
            strategy = VirusLLMStrategy(
                llm_client, verbose=args.verbose, config=llm_config,
                logger=logger,
            )
            strategy.async_client = async_llm_client
        else:
            if i < num_agents * args.llm_ratio:
                strategy = VirusLLMStrategy(
                    llm_client, verbose=args.verbose, config=llm_config,
                    logger=logger,
                )
                strategy.async_client = async_llm_client
            else:
                strategy = VirusHeuristicStrategy(model, logger=logger)

        VirusAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            state=state,
            virus_spread_chance=args.virus_spread_chance,
            virus_check_frequency=args.virus_check_frequency,
            recovery_chance=args.recovery_chance,
            gain_resistance_chance=args.gain_resistance_chance,
        )

    print(f"Created {len(model.agents)} agents")

    model.datacollector.collect(model)

    num_steps = args.steps
    print(f"\nRunning simulation for {num_steps} steps...")
    print("-" * 60)
    print(
        f"{'Step':<8} {'Susceptible':<14} {'Infected':<12} "
        f"{'Resistant':<12} {'Pct Infected':<14}"
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
            print(
                f"done (I={int(latest['Infected'])}, "
                f"R={int(latest['Resistant'])})"
            )
        elif step % 5 == 0 or step == num_steps - 1:
            print(
                f"{step:<8} {int(latest['Susceptible']):<14} "
                f"{int(latest['Infected']):<12} "
                f"{int(latest['Resistant']):<12} "
                f"{latest['Pct Infected']:<14.1f}"
            )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 60)
    print("\nSimulation complete!")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Average time per step: {elapsed_time / num_steps:.3f} seconds")

    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        final_metrics = df.iloc[-1]

        print("\n" + "=" * 60)
        print("FINAL METRICS:")
        print("=" * 60)
        print(f"  Agent Count: {int(final_metrics['Agent Count'])}")
        print(f"  Susceptible: {int(final_metrics['Susceptible'])}")
        print(f"  Infected: {int(final_metrics['Infected'])}")
        print(f"  Resistant: {int(final_metrics['Resistant'])}")
        print(f"  Pct Infected: {final_metrics['Pct Infected']:.1f}%")
        print(f"  Attack Rate: {final_metrics['Attack Rate']:.1f}%")

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
