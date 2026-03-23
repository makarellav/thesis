import json
import time
from typing import Any


class DecisionLogger:

    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        self.records: list[dict[str, Any]] = []
        self.step_counter: int = 0

    def log_decision(
        self,
        agent_id: int,
        agent_type: str,
        context: dict[str, Any],
        action: dict[str, Any],
        strategy: str = "heuristic",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        reasoning: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        record: dict[str, Any] = {
            "step": self.step_counter,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "strategy": strategy,
            "context": context,
            "action": action,
        }

        if strategy == "llm":
            record["system_prompt"] = system_prompt
            record["user_prompt"] = user_prompt
            record["reasoning"] = reasoning
            record["latency_ms"] = round(latency_ms, 1)

        self.records.append(record)

    def log_error(
        self,
        agent_id: int,
        error_msg: str,
        fallback_action: dict[str, Any],
    ) -> None:
        self.records.append({
            "step": self.step_counter,
            "agent_id": agent_id,
            "strategy": "llm",
            "error": error_msg,
            "fallback_action": fallback_action,
        })

    def save(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)

    def get_summary(self) -> dict[str, Any]:
        total = len(self.records)
        errors = sum(1 for r in self.records if "error" in r)
        decisions = [r for r in self.records if "error" not in r]

        llm_decisions = [r for r in decisions if r["strategy"] == "llm"]
        heuristic_decisions = [
            r for r in decisions if r["strategy"] == "heuristic"
        ]
        action_dist: dict[str, int] = {}
        for r in decisions:
            for key, val in r["action"].items():
                label = f"{key}={val}"
                action_dist[label] = action_dist.get(label, 0) + 1

        return {
            "total_records": total,
            "decisions": len(decisions),
            "errors": errors,
            "llm_decisions": len(llm_decisions),
            "heuristic_decisions": len(heuristic_decisions),
            "action_distribution": action_dist,
        }


def make_serializable_context(
    context: dict[str, Any],
) -> dict[str, Any]:
    skip_keys = {"cell", "grid", "neighbors"}
    result: dict[str, Any] = {}
    for key, val in context.items():
        if key in skip_keys:
            continue
        if isinstance(val, (str, int, float, bool, type(None))):
            result[key] = val
        elif isinstance(val, (list, tuple)):
            result[key] = _serialize_list(val)
        elif isinstance(val, dict):
            result[key] = make_serializable_context(val)
        else:
            result[key] = str(val)
    return result


def _serialize_list(items: list[Any] | tuple[Any, ...]) -> list[Any]:
    result: list[Any] = []
    for item in items:
        if isinstance(item, (str, int, float, bool, type(None))):
            result.append(item)
        elif isinstance(item, dict):
            result.append(make_serializable_context(item))
        elif isinstance(item, (list, tuple)):
            result.append(_serialize_list(item))
        else:
            result.append(str(item))
    return result


def measure_latency() -> float:
    return time.time() * 1000
