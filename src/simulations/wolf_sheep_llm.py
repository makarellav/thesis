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
class WolfSheepLLMConfig:
    prompt_style: str = "short"
    include_reasoning: bool = False
    use_async: bool = True
    max_tokens: int = 30
    temperature: float = 0.1
    model: str = "gpt-4o-mini"

    @classmethod
    def baseline(cls, temperature: float = 0.1) -> "WolfSheepLLMConfig":
        return cls(
            prompt_style="long",
            include_reasoning=True,
            use_async=False,
            max_tokens=150,
            temperature=temperature,
        )

    @classmethod
    def optimized(cls, temperature: float = 0.1) -> "WolfSheepLLMConfig":
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
    ) -> "WolfSheepLLMConfig":
        return cls(
            prompt_style="short",
            include_reasoning=True,
            use_async=True,
            max_tokens=200,
            temperature=temperature,
        )


class Direction(str, Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"
    STAY = "STAY"


DIRECTION_OFFSETS: dict[str, tuple[int, int]] = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "STAY": (0, 0),
}


class WolfSheepAction(BaseModel):
    direction: Direction = Field(description="Direction to move (N/E/S/W/STAY)")
    eat: bool = Field(description="Whether to eat if food available")
    reproduce: bool = Field(description="Whether to reproduce")


class WolfSheepActionWithReasoning(BaseModel):
    reasoning: str = Field(description="Brief explanation of decision")
    direction: Direction = Field(description="Direction to move (N/E/S/W/STAY)")
    eat: bool = Field(description="Whether to eat if food available")
    reproduce: bool = Field(description="Whether to reproduce")


class WolfSheepLLMStrategy(LLMStrategy):
    def __init__(
        self,
        client: Any,
        verbose: bool = False,
        config: WolfSheepLLMConfig | None = None,
        logger: DecisionLogger | None = None,
    ) -> None:
        super().__init__(client)
        self.verbose = verbose
        self.config = config if config is not None else WolfSheepLLMConfig.optimized()
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
        agent_type = context["agent_type"]
        grid_width = context["grid_width"]
        grid_height = context["grid_height"]

        if agent_type == "sheep":
            system_prompt, user_prompt = self._build_sheep_prompt(context)
        else:
            system_prompt, user_prompt = self._build_wolf_prompt(context)

        response_model = (
            WolfSheepActionWithReasoning
            if self.config.include_reasoning
            else WolfSheepAction
        )

        try:
            start_ms = measure_latency()
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

            action = {
                "move": new_pos,
                "eat": response.eat,
                "reproduce": response.reproduce,
            }

            elapsed_ms = measure_latency() - start_ms

            if self.logger is not None:
                reasoning_text = (
                    response.reasoning
                    if hasattr(response, "reasoning") else None
                )
                self.logger.log_decision(
                    agent_id=context.get("agent_id", 0),
                    agent_type=agent_type,
                    context=make_serializable_context(context),
                    action=action,
                    strategy="llm",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    reasoning=reasoning_text,
                    latency_ms=elapsed_ms,
                )

            if self.verbose:
                agent_id = context.get("agent_id", "?")
                eat_str = "eat" if response.eat else "skip"
                repro_str = "repro" if response.reproduce else "no-repro"
                if (
                    self.config.include_reasoning
                    and hasattr(response, "reasoning")
                ):
                    print(
                        f"Agent {agent_id} [{agent_type}] → {new_pos} "
                        f"{eat_str} {repro_str} "
                        f"(reasoning: {response.reasoning})"
                    )
                else:
                    print(
                        f"Agent {agent_id} [{agent_type}] → {new_pos} "
                        f"{eat_str} {repro_str}"
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
            fallback = {"move": current_pos, "eat": False, "reproduce": False}
            if self.logger is not None:
                self.logger.log_error(
                    agent_id=context.get("agent_id", 0),
                    error_msg=f"{type(e).__name__}: {e}",
                    fallback_action=fallback,
                )
            return fallback

    def _build_sheep_prompt(
        self, context: dict[str, Any]
    ) -> tuple[str, str]:
        energy = context["energy"]
        wolves_count = context["wolves_nearby_count"]
        grass = context["grass_available"]

        if self.config.prompt_style == "long":
            system_prompt = (
                "You are a sheep in a predator-prey simulation. "
                f"Your energy is {energy:.0f}. You lose 1 energy per step. "
                "You die if energy reaches 0. "
                "You can eat grass to gain energy. "
                "Wolves eat sheep — avoid them! "
                "Choose direction (N/E/S/W/STAY), eat (true/false if grass "
                "available), reproduce (true/false, costs half energy)."
            )
            wolves_info = context["wolves_nearby"]
            grass_info = context["grass_neighbors"]
            user_prompt = (
                f"Energy: {energy:.0f}\n"
                f"Position: {context['position']}\n"
                f"Wolves nearby: {wolves_count} {wolves_info}\n"
                f"Grass in cell: {grass}\n"
                f"Grass neighbors: {grass_info}\n"
                f"Other sheep in cell: {context['sheep_in_cell']}\n"
                "Choose direction, eat, reproduce."
            )
        else:
            system_prompt = (
                f"Prey agent. Energy:{energy:.0f}. "
                "Move(N/E/S/W/STAY), eat(bool), reproduce(bool). "
                "Avoid wolves. Eat grass. Survive."
            )
            user_prompt = (
                f"E:{energy:.0f} Wolves:{wolves_count} "
                f"Grass:{grass} Sheep:{context['sheep_in_cell']}"
            )

        return system_prompt, user_prompt

    def _build_wolf_prompt(
        self, context: dict[str, Any]
    ) -> tuple[str, str]:
        energy = context["energy"]
        sheep_nearby_count = context["sheep_nearby_count"]
        sheep_in_cell = context["sheep_in_cell"]

        if self.config.prompt_style == "long":
            system_prompt = (
                "You are a wolf in a predator-prey simulation. "
                f"Your energy is {energy:.0f}. You lose 1 energy per step. "
                "You die if energy reaches 0. "
                "You can eat sheep to gain energy. "
                "Choose direction (N/E/S/W/STAY), eat (true/false if sheep "
                "in cell), reproduce (true/false, costs half energy)."
            )
            sheep_info = context["sheep_nearby"]
            user_prompt = (
                f"Energy: {energy:.0f}\n"
                f"Position: {context['position']}\n"
                f"Sheep nearby: {sheep_nearby_count} {sheep_info}\n"
                f"Sheep in cell: {sheep_in_cell}\n"
                f"Other wolves in cell: {context['wolves_in_cell']}\n"
                "Choose direction, eat, reproduce."
            )
        else:
            system_prompt = (
                f"Predator agent. Energy:{energy:.0f}. "
                "Move(N/E/S/W/STAY), eat(bool), reproduce(bool). "
                "Hunt sheep. Eat to survive."
            )
            user_prompt = (
                f"E:{energy:.0f} Sheep_near:{sheep_nearby_count} "
                f"Sheep_here:{sheep_in_cell} "
                f"Wolves:{context['wolves_in_cell']}"
            )

        return system_prompt, user_prompt
