import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.strategy import LLMStrategy
from src.utils.decision_logger import (
    DecisionLogger,
    make_serializable_context,
    measure_latency,
)


@dataclass
class VirusLLMConfig:
    prompt_style: str = "short"
    include_reasoning: bool = False
    use_async: bool = True
    max_tokens: int = 30
    temperature: float = 0.1
    model: str = "gpt-4o-mini"

    @classmethod
    def baseline(cls, temperature: float = 0.1) -> "VirusLLMConfig":
        return cls(
            prompt_style="long",
            include_reasoning=True,
            use_async=False,
            max_tokens=150,
            temperature=temperature,
        )

    @classmethod
    def optimized(cls, temperature: float = 0.1) -> "VirusLLMConfig":
        return cls(
            prompt_style="short",
            include_reasoning=False,
            use_async=True,
            max_tokens=30,
            temperature=temperature,
        )

    @classmethod
    def optimized_with_reasoning(
        cls, temperature: float = 0.1
    ) -> "VirusLLMConfig":
        return cls(
            prompt_style="short",
            include_reasoning=True,
            use_async=True,
            max_tokens=200,
            temperature=temperature,
        )


class Direction(str, Enum):
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"
    STAY = "STAY"


DIRECTION_OFFSETS: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "NE": (1, 1),
    "E": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "W": (-1, 0),
    "NW": (-1, 1),
    "STAY": (0, 0),
}


class VirusAction(BaseModel):
    direction: Direction = Field(description="Direction to move")
    interact: bool = Field(description="Whether to interact with neighbors")


class VirusActionWithReasoning(BaseModel):
    reasoning: str = Field(description="Brief explanation of decision")
    direction: Direction = Field(description="Direction to move")
    interact: bool = Field(description="Whether to interact with neighbors")


class VirusLLMStrategy(LLMStrategy):
    def __init__(
        self,
        client: Any,
        verbose: bool = False,
        config: VirusLLMConfig | None = None,
        logger: DecisionLogger | None = None,
    ) -> None:
        super().__init__(client)
        self.verbose = verbose
        self.config = config if config is not None else VirusLLMConfig.optimized()
        self.async_client: Any = None
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use decide_async in async context")
        except RuntimeError:
            return asyncio.run(self.decide_async(context))

    async def decide_async(self, context: dict[str, Any]) -> dict[str, Any]:
        current_pos = context["position"]
        state = context["state"]
        infected_neighbors = context["infected_neighbors"]
        susceptible_neighbors = context["susceptible_neighbors"]
        resistant_neighbors = context["resistant_neighbors"]
        total_neighbors = context["total_neighbors"]
        infection_risk = context["infection_risk"]
        grid_width = context["grid_width"]
        grid_height = context["grid_height"]

        if self.config.prompt_style == "long":
            system_prompt = (
                "You are an agent in a virus spread simulation. "
                f"Your current state is {state}. "
                "Each turn, you move to a neighboring cell and decide "
                "whether to interact with others. "
                "If you are SUSCEPTIBLE, avoid infected neighbors. "
                "If you are INFECTED, consider self-isolating (don't interact) "
                "to reduce spread. "
                "If you are RESISTANT, you are immune and can move freely. "
                "Choose a direction (N/NE/E/SE/S/SW/W/NW/STAY) and whether "
                "to interact."
            )

            user_prompt = (
                f"Your state: {state}\n"
                f"Position: {current_pos}\n"
                f"Infected neighbors: {infected_neighbors}\n"
                f"Susceptible neighbors: {susceptible_neighbors}\n"
                f"Resistant neighbors: {resistant_neighbors}\n"
                f"Total neighbors: {total_neighbors}\n"
                f"Infection risk: {infection_risk:.2f}\n\n"
                "Choose direction and whether to interact."
            )
        else:
            system_prompt = (
                f"Virus sim agent. State:{state}. "
                "Move (N/NE/E/SE/S/SW/W/NW/STAY) and interact(true/false). "
                "Susceptible: avoid infected. Infected: self-isolate."
            )

            user_prompt = (
                f"State:{state} Pos:{current_pos} "
                f"Inf:{infected_neighbors} Sus:{susceptible_neighbors} "
                f"Res:{resistant_neighbors} Risk:{infection_risk:.2f}"
            )

        response_model = (
            VirusActionWithReasoning
            if self.config.include_reasoning
            else VirusAction
        )

        start_ms = measure_latency()
        try:
            if self.config.use_async:
                client = self.async_client if self.async_client else self.client
                response = await client.chat.completions.create(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

            dx, dy = DIRECTION_OFFSETS[response.direction.value]
            new_x = (current_pos[0] + dx) % grid_width
            new_y = (current_pos[1] + dy) % grid_height
            new_pos = (new_x, new_y)

            action = {"move": new_pos, "interact": response.interact}

            if self.verbose:
                agent_id = context.get("agent_id", "?")
                interact_str = "interact" if response.interact else "isolate"
                if self.config.include_reasoning and hasattr(response, "reasoning"):
                    print(
                        f"Agent {agent_id} [{state}] → {new_pos} "
                        f"{interact_str} "
                        f"(reasoning: {response.reasoning})"
                    )
                else:
                    print(
                        f"Agent {agent_id} [{state}] → {new_pos} "
                        f"{interact_str}"
                    )

            if self.logger is not None:
                elapsed_ms = measure_latency() - start_ms
                reasoning_text = (
                    getattr(response, "reasoning", None)
                    if self.config.include_reasoning
                    else None
                )
                self.logger.log_decision(
                    agent_id=context.get("agent_id", 0),
                    agent_type=state,
                    context=make_serializable_context(context),
                    action=action,
                    strategy="llm",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    reasoning=reasoning_text,
                    latency_ms=elapsed_ms,
                )

            return action

        except Exception as e:
            import sys
            import traceback

            print(
                f"\n[LLM ERROR] {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            fallback = {"move": current_pos, "interact": False}
            if self.logger is not None:
                self.logger.log_error(
                    agent_id=context.get("agent_id", 0),
                    error_msg=f"{type(e).__name__}: {e}",
                    fallback_action=fallback,
                )
            return fallback
