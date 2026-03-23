import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "figure.autolayout": True,
})

PROFILES = ["heuristic", "optimized", "optimized_with_reasoning"]
SEEDS = [456, 789, 1337]

PROFILE_LABELS = {
    "heuristic": "Heuristic",
    "optimized": "LLM (Optimized)",
    "optimized_with_reasoning": "LLM (+ Reasoning)",
}
PROFILE_COLORS = {
    "heuristic": "#2196F3",
    "optimized": "#FF9800",
    "optimized_with_reasoning": "#4CAF50",
}

SIMULATION_METRICS: dict[str, list[dict[str, str]]] = {
    "sugarscape": [
        {"column": "Agent Count", "label": "Agent Count (Survival)"},
        {"column": "Gini Coefficient", "label": "Gini Coefficient"},
        {"column": "Average Wealth", "label": "Average Wealth"},
        {"column": "Average Lifespan", "label": "Average Lifespan"},
        {"column": "Suboptimal Move Rate", "label": "Suboptimal Move Rate"},
    ],
    "boltzmann": [
        {"column": "Gini Coefficient", "label": "Gini Coefficient"},
        {"column": "Average Wealth", "label": "Average Wealth"},
        {"column": "Wealth Std Dev", "label": "Wealth Std Dev"},
        {"column": "Suboptimal Trade Rate", "label": "Suboptimal Trade Rate"},
    ],
    "schelling": [
        {"column": "Pct Happy", "label": "% Happy Agents"},
        {"column": "Segregation Index", "label": "Segregation Index"},
        {"column": "Num Unhappy", "label": "Unhappy Agents"},
    ],
    "virus": [
        {"column": "Infected", "label": "Infected"},
        {"column": "Susceptible", "label": "Susceptible"},
        {"column": "Resistant", "label": "Resistant"},
        {"column": "Pct Infected", "label": "% Infected"},
    ],
    "wolfsheep": [
        {"column": "Wolves", "label": "Wolf Population"},
        {"column": "Sheep", "label": "Sheep Population"},
        {"column": "Grass", "label": "Grass Patches"},
        {"column": "Total Population", "label": "Total Population"},
    ],
}


def load_csvs(
    results_dir: Path,
    profiles: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, dict[int, pd.DataFrame]]:
    profiles = profiles or PROFILES
    seeds = seeds or SEEDS

    data: dict[str, dict[int, pd.DataFrame]] = {}
    for profile in profiles:
        data[profile] = {}
        for seed in seeds:
            csv_path = results_dir / f"{profile}_seed{seed}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                data[profile][seed] = df
    return data


def plot_timeseries_comparison(
    data: dict[str, dict[int, pd.DataFrame]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots()

    for profile in PROFILES:
        if profile not in data or not data[profile]:
            continue

        dfs = list(data[profile].values())
        if not dfs:
            continue

        min_len = min(len(df) for df in dfs)
        if metric not in dfs[0].columns:
            continue

        values = np.array([df[metric].values[:min_len] for df in dfs])
        steps = np.arange(min_len)

        mean = values.mean(axis=0)
        vmin = values.min(axis=0)
        vmax = values.max(axis=0)

        color = PROFILE_COLORS[profile]
        label = PROFILE_LABELS[profile]

        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, vmin, vmax, alpha=0.15, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_final_metrics_bar(
    data: dict[str, dict[int, pd.DataFrame]],
    metrics: list[dict[str, str]],
    title: str,
    output_path: Path,
) -> None:
    available_metrics = [
        m for m in metrics
        if any(
            m["column"] in df.columns
            for seed_dfs in data.values()
            for df in seed_dfs.values()
        )
    ]
    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    n_profiles = len([p for p in PROFILES if p in data and data[p]])
    if n_profiles == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, m in zip(axes, available_metrics, strict=True):
        means = []
        stds = []
        labels = []
        colors = []

        for profile in PROFILES:
            if profile not in data or not data[profile]:
                continue

            finals = []
            for df in data[profile].values():
                if m["column"] in df.columns and not df.empty:
                    finals.append(df[m["column"]].iloc[-1])

            if finals:
                means.append(np.mean(finals))
                stds.append(np.std(finals))
                labels.append(PROFILE_LABELS[profile])
                colors.append(PROFILE_COLORS[profile])

        if means:
            x = np.arange(len(means))
            ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(m["label"])
            ax.set_title(m["label"])
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_boxplots(
    data: dict[str, dict[int, pd.DataFrame]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = []
    labels = []
    for profile in PROFILES:
        if profile not in data or not data[profile]:
            continue
        finals = []
        for df in data[profile].values():
            if metric in df.columns and not df.empty:
                finals.append(df[metric].iloc[-1])
        if finals:
            box_data.append(finals)
            labels.append(PROFILE_LABELS[profile])

    if not box_data:
        plt.close(fig)
        return

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    colors = [PROFILE_COLORS[p] for p in PROFILES if p in data and data[p]]
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def analyze_decision_logs(
    results_dir: Path,
    simulation: str,
    output_dir: Path,
) -> None:
    print(f"\nAnalyzing decision logs for {simulation}...")

    for profile in PROFILES:
        all_records: list[dict[str, Any]] = []
        for seed in SEEDS:
            json_path = results_dir / f"{profile}_seed{seed}_decisions.json"
            if json_path.exists():
                with open(json_path) as f:
                    records = json.load(f)
                    all_records.extend(records)

        if not all_records:
            continue

        decisions = [r for r in all_records if "error" not in r]
        errors = [r for r in all_records if "error" in r]

        action_counter: Counter[str] = Counter()
        for r in decisions:
            for key, val in r["action"].items():
                action_counter[f"{key}={val}"] += 1

        if action_counter:
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_actions = action_counter.most_common(15)
            labels_a = [a[0] for a in sorted_actions]
            counts = [a[1] for a in sorted_actions]

            color = PROFILE_COLORS.get(profile, "#888")
            ax.barh(range(len(labels_a)), counts, color=color, alpha=0.8)
            ax.set_yticks(range(len(labels_a)))
            ax.set_yticklabels(labels_a)
            ax.set_xlabel("Count")
            ax.set_title(
                f"{simulation.title()} - Action Distribution "
                f"({PROFILE_LABELS[profile]})"
            )
            ax.grid(True, alpha=0.3, axis="x")
            ax.invert_yaxis()

            path = output_dir / f"{simulation}_actions_{profile}.png"
            fig.savefig(path)
            plt.close(fig)
            print(f"  Saved: {path}")

        if profile != "heuristic":
            latencies = [
                r["latency_ms"] for r in decisions
                if r.get("strategy") == "llm" and "latency_ms" in r
            ]
            if latencies:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(latencies, bins=30, color=PROFILE_COLORS[profile],
                        alpha=0.7, edgecolor="white")
                ax.axvline(np.mean(latencies), color="red", linestyle="--",
                           label=f"Mean: {np.mean(latencies):.0f}ms")
                ax.axvline(np.median(latencies), color="orange", linestyle="--",
                           label=f"Median: {np.median(latencies):.0f}ms")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Count")
                ax.set_title(
                    f"{simulation.title()} - LLM Latency "
                    f"({PROFILE_LABELS[profile]})"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                path = output_dir / f"{simulation}_latency_{profile}.png"
                fig.savefig(path)
                plt.close(fig)
                print(f"  Saved: {path}")

        print(f"\n  {PROFILE_LABELS[profile]}:")
        print(f"    Total decisions: {len(decisions)}")
        print(f"    Errors: {len(errors)}")
        if profile != "heuristic":
            latencies = [
                r["latency_ms"] for r in decisions
                if r.get("strategy") == "llm" and "latency_ms" in r
            ]
            if latencies:
                print(f"    Avg latency: {np.mean(latencies):.0f}ms")
                print(f"    P95 latency: {np.percentile(latencies, 95):.0f}ms")
        top3 = action_counter.most_common(3)
        if top3:
            print(f"    Top actions: {', '.join(f'{a}({c})' for a, c in top3)}")


def analyze_simulation(
    simulation: str,
    results_dir: Path,
    output_dir: Path,
    decisions: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Analyzing: {simulation.title()}")
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    data = load_csvs(results_dir)

    loaded = sum(len(seeds) for seeds in data.values())
    if loaded == 0:
        print(f"  No CSV files found in {results_dir}")
        print("  Run experiments with --output-csv first.")
        return

    print(f"  Loaded {loaded} CSV files")
    for profile, seeds in data.items():
        if seeds:
            print(f"    {PROFILE_LABELS[profile]}: {len(seeds)} seeds")

    metrics = SIMULATION_METRICS.get(simulation, [])

    print("\nGenerating time-series plots...")
    for m in metrics:
        plot_timeseries_comparison(
            data,
            metric=m["column"],
            ylabel=m["label"],
            title=f"{simulation.title()} - {m['label']} Over Time",
            output_path=output_dir / (
                f"{simulation}_ts_"
                f"{m['column'].lower().replace(' ', '_')}.png"
            ),
        )

    print("\nGenerating final metrics bar chart...")
    plot_final_metrics_bar(
        data,
        metrics=metrics,
        title=f"{simulation.title()} - Final Metrics Comparison",
        output_path=output_dir / f"{simulation}_final_metrics.png",
    )

    print("\nGenerating box plots...")
    for m in metrics[:3]:
        plot_boxplots(
            data,
            metric=m["column"],
            ylabel=m["label"],
            title=f"{simulation.title()} - {m['label']} (Final Values)",
            output_path=output_dir / (
                f"{simulation}_box_"
                f"{m['column'].lower().replace(' ', '_')}.png"
            ),
        )

    if decisions:
        analyze_decision_logs(results_dir, simulation, output_dir)


def find_results_dirs() -> dict[str, list[Path]]:
    root = Path(".")
    found: dict[str, list[Path]] = {}

    sim_prefixes = {
        "sugarscape": "results_multiseed_",
        "boltzmann": "results_boltzmann_",
        "schelling": "results_schelling_",
        "virus": "results_virus_",
        "wolfsheep": "results_wolfsheep_",
    }

    for sim_name, prefix in sim_prefixes.items():
        dirs = sorted(root.glob(f"{prefix}*"))
        if dirs:
            found[sim_name] = dirs

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze simulation results and generate graphs"
    )
    parser.add_argument(
        "--simulation",
        choices=["sugarscape", "boltzmann", "schelling", "virus", "wolfsheep"],
        help="Simulation to analyze",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Path to results directory containing CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Auto-detect and analyze all available results",
    )
    parser.add_argument(
        "--decisions",
        action="store_true",
        help="Also analyze decision log JSON files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all:
        found = find_results_dirs()
        if not found:
            print("No results directories found.")
            print("Run experiments first, e.g.:")
            print("  bash run_boltzmann_multiseed.sh")
            return

        total = sum(len(dirs) for dirs in found.values())
        print(f"Found {total} results directories:")
        for sim, dirs in found.items():
            for d in dirs:
                print(f"  {sim}: {d}")

        for sim, dirs in found.items():
            for d in dirs:
                analyze_simulation(
                    sim, d,
                    output_dir / d.name,
                    decisions=args.decisions,
                )
    elif args.simulation and args.results_dir:
        results_path = Path(args.results_dir)
        analyze_simulation(
            args.simulation,
            results_path,
            output_dir / results_path.name,
            decisions=args.decisions,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  uv run python -m src.utils.analyze_results --all")
        print("  uv run python -m src.utils.analyze_results "
              "--simulation boltzmann "
              "--results-dir results_boltzmann_gpt_4o_mini_temp0_1")


if __name__ == "__main__":
    main()
