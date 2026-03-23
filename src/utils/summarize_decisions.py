import argparse
import contextlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROFILES = ["heuristic", "optimized", "optimized_with_reasoning"]
SEEDS = [456, 789, 1337]

PROFILE_LABELS = {
    "heuristic": "Heuristic",
    "optimized": "LLM (Optimized)",
    "optimized_with_reasoning": "LLM (+ Reasoning)",
}

STATIC_CONTEXT_KEYS = {
    "agent_id", "metabolism_sugar", "metabolism_spice", "vision",
    "virus_spread_chance", "virus_check_frequency",
    "recovery_chance", "gain_resistance_chance",
    "grid_width", "grid_height",
}


def load_decision_logs(
    results_dir: Path,
    profiles: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    profiles = profiles or PROFILES
    seeds = seeds or SEEDS

    data: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        records: list[dict[str, Any]] = []
        for seed in seeds:
            path = results_dir / f"{profile}_seed{seed}_decisions.json"
            if path.exists():
                with open(path) as f:
                    records.extend(json.load(f))
        if records:
            data[profile] = records
    return data


def compute_move_stats(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = [r for r in records if "error" not in r]
    dists: list[int] = []
    stayed = 0
    for r in decisions:
        ctx = r.get("context", {})
        pos = ctx.get("position")
        action = r.get("action", {})
        move = action.get("move")
        if pos and move and isinstance(pos, list) and isinstance(move, list):
            d = abs(move[0] - pos[0]) + abs(move[1] - pos[1])
            dists.append(d)
            if d == 0:
                stayed += 1

    if not dists:
        return {}

    arr = np.array(dists)
    return {
        "mean_distance": round(float(arr.mean()), 2),
        "median_distance": int(np.median(arr)),
        "max_distance": int(arr.max()),
        "stayed_pct": round(100 * stayed / len(dists), 1),
        "unique_targets": len({
            tuple(r["action"]["move"])
            for r in decisions
            if "move" in r.get("action", {})
        }),
    }


def compute_population_over_time(
    records: list[dict[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    decisions = [r for r in records if "error" not in r]
    by_step: dict[int, set[int]] = defaultdict(set)
    for r in decisions:
        step = r.get("step", 0)
        aid = r.get("agent_id", 0)
        by_step[step].add(aid)

    n_early = max_steps // 3
    n_mid = 2 * max_steps // 3

    def avg_agents(start: int, end: int) -> float:
        counts = [
            len(by_step.get(s, set()))
            for s in range(start, end)
        ]
        return round(float(np.mean(counts)), 1) if counts else 0.0

    return {
        f"early (0-{n_early})": avg_agents(0, n_early),
        f"mid ({n_early}-{n_mid})": avg_agents(n_early, n_mid),
        f"late ({n_mid}-{max_steps})": avg_agents(n_mid, max_steps),
    }


def compute_dynamic_context_stats(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    decisions = [r for r in records if "error" not in r]
    if not decisions:
        return {}

    sample_ctx = decisions[0].get("context", {})
    dynamic_keys = [
        k for k, v in sample_ctx.items()
        if k not in STATIC_CONTEXT_KEYS
        and isinstance(v, (int, float))
        and k != "agent_id"
    ]

    for r in decisions:
        ctx = r.get("context", {})
        for k in list(ctx.keys()):
            if (
                k not in STATIC_CONTEXT_KEYS
                and k not in dynamic_keys
                and k != "agent_id"
                and isinstance(ctx.get(k), (int, float, str))
            ):
                try:
                    float(ctx[k])
                    dynamic_keys.append(k)
                except (ValueError, TypeError):
                    pass
        break

    result: dict[str, dict[str, float]] = {}
    for key in dynamic_keys:
        values = []
        for r in decisions:
            val = r.get("context", {}).get(key)
            if val is not None:
                with contextlib.suppress(ValueError, TypeError):
                    values.append(float(val))
        if values:
            arr = np.array(values)
            result[key] = {
                "mean": round(float(arr.mean()), 2),
                "std": round(float(arr.std()), 2),
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
            }
    return result


def summarize_latency(
    records: list[dict[str, Any]],
) -> dict[str, float] | None:
    latencies = [
        r["latency_ms"] for r in records
        if r.get("strategy") == "llm" and "latency_ms" in r
    ]
    if not latencies:
        return None
    arr = np.array(latencies)
    return {
        "count": len(latencies),
        "mean_ms": round(float(arr.mean()), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
    }


def summarize_errors(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [r for r in records if "error" in r]
    if not errors:
        return {"count": 0}

    error_msgs: Counter[str] = Counter()
    for e in errors:
        msg = e.get("error", "unknown")
        error_msgs[msg[:100]] += 1

    return {
        "count": len(errors),
        "types": dict(error_msgs.most_common(5)),
    }


def extract_diverse_reasoning_samples(
    records: list[dict[str, Any]],
    n_samples: int = 5,
) -> list[dict[str, Any]]:
    llm_records = [
        r for r in records
        if r.get("strategy") == "llm"
        and "error" not in r
        and r.get("reasoning")
    ]
    if not llm_records:
        return []

    max_step = max(r.get("step", 0) for r in llm_records)
    buckets: list[list[dict[str, Any]]] = [
        [] for _ in range(n_samples)
    ]
    for r in llm_records:
        step = r.get("step", 0)
        idx = min(
            int(step / (max_step + 1) * n_samples),
            n_samples - 1,
        )
        buckets[idx].append(r)

    samples = []
    seen_reasonings: set[str] = set()
    for bucket in buckets:
        if not bucket:
            continue
        bucket.sort(
            key=lambda r: len(r.get("reasoning", "")),
            reverse=True,
        )
        for r in bucket:
            reasoning = r.get("reasoning", "")
            short = reasoning[:50]
            if short not in seen_reasonings:
                seen_reasonings.add(short)
                samples.append(_extract_sample(r))
                break

    return samples[:n_samples]


def _extract_sample(record: dict[str, Any]) -> dict[str, Any]:
    ctx = record.get("context", {})
    dynamic_ctx = {
        k: v for k, v in ctx.items()
        if k not in STATIC_CONTEXT_KEYS
    }
    return {
        "step": record.get("step"),
        "agent_id": record.get("agent_id"),
        "agent_type": record.get("agent_type"),
        "context": dynamic_ctx,
        "action": record.get("action"),
        "reasoning": record.get("reasoning"),
    }


def build_profile_summary(
    profile: str,
    records: list[dict[str, Any]],
    max_steps: int | None = None,
) -> dict[str, Any]:
    decisions = [r for r in records if "error" not in r]
    if not decisions:
        return {"profile": profile, "total_records": 0}

    if max_steps is None:
        max_steps = max(r.get("step", 0) for r in decisions) + 1

    summary: dict[str, Any] = {
        "profile": PROFILE_LABELS.get(profile, profile),
        "total_decisions": len(decisions),
        "total_errors": len([r for r in records if "error" in r]),
        "steps_covered": max_steps,
        "population_over_time": compute_population_over_time(
            records, max_steps,
        ),
        "move_stats": compute_move_stats(records),
        "dynamic_context": compute_dynamic_context_stats(records),
        "errors": summarize_errors(records),
    }

    latency = summarize_latency(records)
    if latency:
        summary["latency"] = latency

    if profile != "heuristic":
        summary["reasoning_samples"] = (
            extract_diverse_reasoning_samples(records, n_samples=5)
        )

    return summary


def format_summary_markdown(
    simulation: str,
    summaries: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {simulation.title()} — Decision Log Summary",
        "",
    ]

    for s in summaries:
        profile = s.get("profile", "?")
        lines.append(f"## {profile}")
        lines.append("")
        lines.append(
            f"- Decisions: {s.get('total_decisions', 0)} | "
            f"Errors: {s.get('total_errors', 0)} | "
            f"Steps: {s.get('steps_covered', '?')}"
        )

        latency = s.get("latency")
        if latency:
            lines.append(
                f"- Latency: mean={latency['mean_ms']}ms, "
                f"median={latency['median_ms']}ms, "
                f"p95={latency['p95_ms']}ms"
            )
        lines.append("")

        pop = s.get("population_over_time", {})
        if pop:
            lines.append("### Active Agents Over Time")
            for period, avg in pop.items():
                lines.append(f"  - {period}: avg {avg} agents/step")
            lines.append("")

        move = s.get("move_stats", {})
        if move:
            lines.append("### Movement Behavior")
            lines.append(
                f"  - Avg distance: {move['mean_distance']} cells "
                f"(median: {move['median_distance']}, "
                f"max: {move['max_distance']})"
            )
            lines.append(
                f"  - Stayed in place: {move['stayed_pct']}%"
            )
            lines.append(
                f"  - Unique cells targeted: "
                f"{move['unique_targets']}"
            )
            lines.append("")

        ctx = s.get("dynamic_context", {})
        if ctx:
            lines.append("### Agent State at Decision Time")
            for key, stats in ctx.items():
                lines.append(
                    f"  - {key}: mean={stats['mean']}, "
                    f"std={stats['std']}, "
                    f"range=[{stats['min']}, {stats['max']}]"
                )
            lines.append("")

        samples = s.get("reasoning_samples", [])
        if samples:
            lines.append("### Reasoning Samples (diverse)")
            for i, sample in enumerate(samples, 1):
                lines.append(
                    f"  **Sample {i}** — "
                    f"step {sample['step']}, "
                    f"agent {sample['agent_id']} "
                    f"({sample.get('agent_type', '?')})"
                )
                ctx_s = sample.get("context", {})
                if ctx_s:
                    ctx_items = [
                        f"{k}={v}" for k, v in ctx_s.items()
                        if k not in ("agent_id",)
                        and not isinstance(v, (list, dict))
                    ]
                    if ctx_items:
                        lines.append(
                            f"  State: {', '.join(ctx_items[:8])}"
                        )
                lines.append(f"  Action: {sample['action']}")
                lines.append(
                    f"  Reasoning: "
                    f"{sample.get('reasoning', 'N/A')}"
                )
                lines.append("")

        errors = s.get("errors", {})
        if errors.get("count", 0) > 0:
            lines.append("### Errors")
            lines.append(f"  Total: {errors['count']}")
            for msg, cnt in errors.get("types", {}).items():
                lines.append(f"  - ({cnt}x) {msg}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


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


def summarize_simulation(
    simulation: str,
    results_dir: Path,
    output_path: Path | None = None,
    figures_dir: Path | None = None,
) -> str:
    print(f"\nSummarizing: {simulation.title()}")
    print(f"  Results dir: {results_dir}")

    if figures_dir is not None:
        from src.utils.analyze_results import analyze_simulation
        analyze_simulation(
            simulation,
            results_dir,
            figures_dir / results_dir.name,
            decisions=True,
        )

    data = load_decision_logs(results_dir)
    if not data:
        msg = f"  No decision log files found in {results_dir}"
        print(msg)
        return msg

    summaries = []
    for profile in PROFILES:
        if profile in data:
            print(
                f"  Processing {profile}: "
                f"{len(data[profile])} records"
            )
            summary = build_profile_summary(
                profile, data[profile],
            )
            summaries.append(summary)

    markdown = format_summary_markdown(simulation, summaries)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown)
        print(f"  Saved: {output_path}")

    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize decision logs into compact markdown "
            "for LLM analysis"
        ),
    )
    parser.add_argument(
        "--simulation",
        choices=[
            "sugarscape", "boltzmann", "schelling",
            "virus", "wolfsheep",
        ],
        help="Simulation to summarize",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help=(
            "Path to results directory containing "
            "*_decisions.json files"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="summaries",
        help=(
            "Output directory for markdown summaries "
            "(default: summaries/)"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help=(
            "Output directory for figures "
            "(default: figures/)"
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Auto-detect and summarize all available results",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_output",
        help="Print summary to stdout (in addition to saving)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = (
        None if args.no_figures else Path(args.figures_dir)
    )

    if args.all:
        found = find_results_dirs()
        if not found:
            print("No results directories found.")
            return

        total = sum(len(dirs) for dirs in found.values())
        print(f"Found {total} results directories:")
        for sim, dirs in found.items():
            for d in dirs:
                print(f"  {sim}: {d}")

        for sim, dirs in found.items():
            for d in dirs:
                suffix = d.name
                md = summarize_simulation(
                    sim, d,
                    output_path=(
                        output_dir
                        / f"{sim}_{suffix}_decisions.md"
                    ),
                    figures_dir=fig_dir,
                )
                if args.print_output:
                    print(md)

    elif args.simulation and args.results_dir:
        results_path = Path(args.results_dir)
        suffix = results_path.name
        md = summarize_simulation(
            args.simulation,
            results_path,
            output_path=(
                output_dir
                / f"{args.simulation}_{suffix}_decisions.md"
            ),
            figures_dir=fig_dir,
        )
        if args.print_output:
            print(md)
    else:
        parser.print_help()
        print("\nExamples:")
        print(
            "  uv run python -m src.utils.summarize_decisions"
            " --all"
        )
        print(
            "  uv run python -m src.utils.summarize_decisions "
            "--simulation sugarscape "
            "--results-dir "
            "results_multiseed_gpt_4o_mini_temp0_0"
        )
        print(
            "  uv run python -m src.utils.summarize_decisions "
            "--simulation sugarscape "
            "--results-dir "
            "results_multiseed_gpt_4o_mini_temp0_0"
            " --no-figures --print"
        )


if __name__ == "__main__":
    main()
