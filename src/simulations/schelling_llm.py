import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.agents.strategy import LLMStrategy
from src.utils.decision_logger import (
    DecisionLogger,
    make_serializable_context,
    measure_latency,
)


@dataclass
class SchellingLLMConfig:
    prompt_style: str = "short"
    include_reasoning: bool = False
    use_async: bool = True
    max_tokens: int = 30
    temperature: float = 0.1
    model: str = "gpt-4o-mini"

    @classmethod
    def baseline(cls, temperature: float = 0.1) -> "SchellingLLMConfig":
        return cls(
            prompt_style="long",
            include_reasoning=True,
            use_async=False,
            max_tokens=150,
            temperature=temperature,
        )

    @classmethod
    def optimized(cls, temperature: float = 0.1) -> "SchellingLLMConfig":
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
    ) -> "SchellingLLMConfig":
        return cls(
            prompt_style="short",
            include_reasoning=True,
            use_async=True,
            max_tokens=200,
            temperature=temperature,
        )


class SchellingAction(BaseModel):
    move: bool = Field(description="Whether to move to a new location")


class SchellingActionWithReasoning(BaseModel):
    reasoning: str = Field(description="Brief explanation of decision")
    move: bool = Field(description="Whether to move to a new location")


class SchellingLLMStrategy(LLMStrategy):
    def __init__(
        self,
        client: Any,
        verbose: bool = False,
        config: SchellingLLMConfig | None = None,
        homophily: float = 0.5,
        logger: DecisionLogger | None = None,
    ) -> None:
        super().__init__(client)
        self.verbose = verbose
        self.config = config if config is not None else SchellingLLMConfig.optimized()
        self.async_client: Any = None
        self.homophily = homophily
        self.logger = logger

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use decide_async in async context")
        except RuntimeError:
            return asyncio.run(self.decide_async(context))

    async def decide_async(self, context: dict[str, Any]) -> dict[str, Any]:
        agent_type = context["agent_type"]
        similar_count = context["similar_count"]
        different_count = context["different_count"]
        total_neighbors = context["total_neighbors"]
        similar_fraction = context["similar_fraction"]
        homophily = context["homophily"]

        if self.config.prompt_style == "long":
            type_label = "A" if agent_type == 0 else "B"
            system_prompt = (
                "You are a resident in a neighborhood segregation simulation. "
                f"You are type {type_label}. You prefer at least "
                f"{homophily:.0%} of your neighbors to be the same type as you. "
                "Each turn, you look at your neighbors and decide whether to "
                "stay in your current location or move to a new random location. "
                "If you are happy with your neighborhood (enough similar neighbors), "
                "you should stay. If you are unhappy, you should move."
            )

            user_prompt = (
                f"Your type: {type_label}\n"
                f"Similar neighbors: {similar_count}\n"
                f"Different neighbors: {different_count}\n"
                f"Total neighbors: {total_neighbors}\n"
                f"Similarity fraction: {similar_fraction:.2f}\n"
                f"Your threshold: {homophily:.2f}\n\n"
                "Should you stay or move?"
            )
        else:
            type_label = "A" if agent_type == 0 else "B"
            system_prompt = (
                f"Schelling agent type {type_label}. "
                f"Threshold: {homophily:.0%} similar neighbors. "
                "Decide: move(true=relocate, false=stay)."
            )

            user_prompt = (
                f"Type:{type_label} "
                f"Similar:{similar_count} Diff:{different_count} "
                f"Frac:{similar_fraction:.2f} Thresh:{homophily:.2f}"
            )

        response_model = (
            SchellingActionWithReasoning
            if self.config.include_reasoning
            else SchellingAction
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

            action = {"move": response.move}

            if self.verbose:
                agent_id = context.get("agent_id", "?")
                move_str = "MOVE" if response.move else "STAY"
                if self.config.include_reasoning and hasattr(response, "reasoning"):
                    print(
                        f"Agent {agent_id} → {move_str} "
                        f"(reasoning: {response.reasoning})"
                    )
                else:
                    print(f"Agent {agent_id} → {move_str}")

            if self.logger is not None:
                elapsed_ms = measure_latency() - start_ms
                reasoning_text = (
                    getattr(response, "reasoning", None)
                    if self.config.include_reasoning
                    else None
                )
                self.logger.log_decision(
                    agent_id=context.get("agent_id", 0),
                    agent_type=str(agent_type),
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
            fallback = {"move": False}
            if self.logger is not None:
                self.logger.log_error(
                    agent_id=context.get("agent_id", 0),
                    error_msg=f"{type(e).__name__}: {e}",
                    fallback_action=fallback,
                )
            return fallback
