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
class BoltzmannLLMConfig:
    prompt_style: str = "short"
    include_reasoning: bool = False
    use_async: bool = True
    max_tokens: int = 30
    temperature: float = 0.1
    model: str = "gpt-4o-mini"

    @classmethod
    def baseline(cls, temperature: float = 0.1) -> "BoltzmannLLMConfig":
        return cls(
            prompt_style="long",
            include_reasoning=True,
            use_async=False,
            max_tokens=150,
            temperature=temperature,
        )

    @classmethod
    def optimized(cls, temperature: float = 0.1) -> "BoltzmannLLMConfig":
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
    ) -> "BoltzmannLLMConfig":
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


class BoltzmannAction(BaseModel):
    direction: Direction = Field(description="Direction to move")
    give: bool = Field(description="Whether to give 1 coin to a cellmate")


class BoltzmannActionWithReasoning(BaseModel):
    reasoning: str = Field(description="Brief explanation of decision")
    direction: Direction = Field(description="Direction to move")
    give: bool = Field(description="Whether to give 1 coin to a cellmate")


class BoltzmannLLMStrategy(LLMStrategy):
    def __init__(
        self,
        client: Any,
        verbose: bool = False,
        config: BoltzmannLLMConfig | None = None,
        logger: DecisionLogger | None = None,
    ) -> None:
        super().__init__(client)
        self.verbose = verbose
        self.config = config if config is not None else BoltzmannLLMConfig.optimized()
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
        wealth = context["wealth"]
        neighbors = context["neighbors"]
        cellmates = context["cellmates"]

        grid_width = context["grid_width"]
        grid_height = context["grid_height"]

        if self.config.prompt_style == "long":
            system_prompt = (
                "You are an agent in a Boltzmann Wealth Model simulation. "
                "You live on a grid where agents exchange wealth. "
                "Each turn, you move to a neighboring cell (Moore neighborhood: "
                "8 adjacent cells or stay). If you have coins (wealth > 0), "
                "you may give 1 coin to a random agent in your new cell. "
                "Your goal is to maximize your wealth over time by making "
                "strategic movement and exchange decisions."
            )

            neighbor_desc = []
            for n in neighbors:
                neighbor_desc.append(
                    f"  Cell {n['position']}: "
                    f"{n['agent_count']} agents, "
                    f"total wealth {n['total_wealth']}"
                )

            cellmate_desc = []
            for c in cellmates:
                cellmate_desc.append(f"  Agent {c['id']}: wealth={c['wealth']}")

            user_prompt = (
                f"Your position: {current_pos}\n"
                f"Your wealth: {wealth} coins\n\n"
                f"Neighboring cells:\n"
                + "\n".join(neighbor_desc)
                + "\n\n"
                "Current cellmates:\n"
                + ("\n".join(cellmate_desc) if cellmate_desc else "  None")
                + "\n\n"
                "Choose a direction to move and whether to give a coin."
            )
        else:
            system_prompt = (
                "Boltzmann wealth agent on grid. Move (N/NE/E/SE/S/SW/W/NW/STAY) "
                "and decide give(true/false). Maximize your wealth."
            )

            neighbor_compact = []
            for n in neighbors:
                neighbor_compact.append(
                    f"{n['position']}:{n['agent_count']}a,{n['total_wealth']}w"
                )

            cellmate_compact = (
                ",".join(f"w{c['wealth']}" for c in cellmates)
                if cellmates
                else "none"
            )

            user_prompt = (
                f"Pos:{current_pos} W:{wealth} "
                f"Neighbors:[{','.join(neighbor_compact)}] "
                f"Cellmates:[{cellmate_compact}]"
            )

        response_model = (
            BoltzmannActionWithReasoning
            if self.config.include_reasoning
            else BoltzmannAction
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

            action = {"move": new_pos, "give": response.give}

            if self.verbose:
                agent_id = context.get("agent_id", "?")
                if self.config.include_reasoning and hasattr(response, "reasoning"):
                    print(
                        f"Agent {agent_id} at {current_pos} → {new_pos} "
                        f"give={response.give} "
                        f"(reasoning: {response.reasoning})"
                    )
                else:
                    print(
                        f"Agent {agent_id} at {current_pos} → {new_pos} "
                        f"give={response.give}"
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
                    agent_type="wealth_agent",
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
            fallback = {"move": current_pos, "give": False}
            if self.logger is not None:
                self.logger.log_error(
                    agent_id=context.get("agent_id", 0),
                    error_msg=f"{type(e).__name__}: {e}",
                    fallback_action=fallback,
                )
            return fallback
