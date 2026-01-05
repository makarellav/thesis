"""LLM-based decision strategy for Sugarscape simulation.

This module implements an LLM-driven strategy that uses natural language
prompts to guide agent movement decisions in the Sugarscape environment.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.agents.strategy import LLMStrategy


@dataclass
class LLMConfig:
    """Configuration for LLM strategy behavior.

    Attributes:
        prompt_style: "short" for compact prompts, "long" for detailed descriptions.
        include_reasoning: Whether to request reasoning field in output.
        use_async: Whether to use async API calls (faster with multiple agents).
        use_cache: Whether to cache decisions for similar contexts.
        max_tokens: Maximum tokens for LLM response.
        temperature: Temperature for sampling (0.0-1.0, lower = more deterministic).
        model: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
    """

    prompt_style: str = "short"
    include_reasoning: bool = False
    use_async: bool = True
    use_cache: bool = False
    max_tokens: int = 30
    temperature: float = 0.1
    model: str = "gpt-4o-mini"

    @classmethod
    def baseline(cls, temperature: float = 0.1) -> "LLMConfig":
        """Create baseline configuration (original, unoptimized approach).

        Args:
            temperature: Temperature for sampling (0.0-1.0).

        Returns:
            LLMConfig with verbose prompts, reasoning, no async, no cache.
        """
        return cls(
            prompt_style="long",
            include_reasoning=True,
            use_async=False,
            use_cache=False,
            max_tokens=150,
            temperature=temperature,
        )

    @classmethod
    def optimized(cls, temperature: float = 0.1) -> "LLMConfig":
        """Create optimized configuration (all optimizations except cache).

        Args:
            temperature: Temperature for sampling (0.0-1.0).

        Returns:
            LLMConfig with short prompts, no reasoning, async, no cache.
        """
        return cls(
            prompt_style="short",
            include_reasoning=False,
            use_async=True,
            use_cache=False,
            max_tokens=30,
            temperature=temperature,
        )

    @classmethod
    def cached(cls, temperature: float = 0.1) -> "LLMConfig":
        """Create cached configuration (optimized + fuzzy caching).

        Args:
            temperature: Temperature for sampling (0.0-1.0).

        Returns:
            LLMConfig with all optimizations including cache.
        """
        return cls(
            prompt_style="short",
            include_reasoning=False,
            use_async=True,
            use_cache=True,
            max_tokens=30,
            temperature=temperature,
        )

    @classmethod
    def optimized_with_reasoning(cls, temperature: float = 0.1) -> "LLMConfig":
        """Create optimized config with chain-of-thought reasoning.

        Args:
            temperature: Temperature for sampling (0.0-1.0).

        Returns:
            LLMConfig with optimizations and reasoning enabled.
        """
        return cls(
            prompt_style="short",
            include_reasoning=True,
            use_async=True,
            use_cache=False,
            max_tokens=200,
            temperature=temperature,
        )


class DecisionCache:
    """Cache for LLM decisions with fuzzy context matching.

    Uses quantized resource values and sorted neighbor options to create
    cache keys that match similar situations, improving hit rate.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize cache with size limit.

        Args:
            max_size: Maximum number of cached decisions.
        """
        self.cache: dict[tuple[Any, ...], tuple[int, int]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _create_key(
        self,
        sugar: int,
        spice: int,
        neighbor_options: list[dict[str, Any]],
    ) -> tuple[Any, ...]:
        """Create fuzzy cache key from context.

        Args:
            sugar: Agent's current sugar level.
            spice: Agent's current spice level.
            neighbor_options: Available neighbor cells with resources.

        Returns:
            Hashable cache key tuple.
        """
        sugar_bucket = sugar // 3
        spice_bucket = spice // 3

        current_pos = neighbor_options[-1]["position"]

        neighbor_key = []
        for opt in neighbor_options:
            if opt["occupied"]:
                continue

            pos = opt["position"]

            dx = pos[0] - current_pos[0]
            dy = pos[1] - current_pos[1]

            sugar_q = opt["sugar"] // 2
            spice_q = opt["spice"] // 2

            neighbor_key.append((dx, dy, sugar_q, spice_q))

        neighbor_key_tuple = tuple(sorted(neighbor_key))

        return (sugar_bucket, spice_bucket, neighbor_key_tuple)

    def get(
        self,
        sugar: int,
        spice: int,
        neighbor_options: list[dict[str, Any]],
    ) -> tuple[int, int] | None:
        """Get cached decision if available.

        Args:
            sugar: Agent's current sugar level.
            spice: Agent's current spice level.
            neighbor_options: Available neighbor cells with resources.

        Returns:
            Cached (x, y) position or None if not cached.
        """
        key = self._create_key(sugar, spice, neighbor_options)
        result = self.cache.get(key)

        if result is not None:
            self.hits += 1
        else:
            self.misses += 1

        return result

    def put(
        self,
        sugar: int,
        spice: int,
        neighbor_options: list[dict[str, Any]],
        decision: tuple[int, int],
    ) -> None:
        """Store decision in cache.

        Args:
            sugar: Agent's current sugar level.
            spice: Agent's current spice level.
            neighbor_options: Available neighbor cells with resources.
            decision: The (x, y) position decision to cache.
        """
        key = self._create_key(sugar, spice, neighbor_options)

        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = decision

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, and hit rate.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class SugarscapeAction(BaseModel):
    """Structured action output from LLM for Sugarscape agents (minimal).

    Attributes:
        move_x: X coordinate where the agent should move.
        move_y: Y coordinate where the agent should move.
    """

    move_x: int = Field(description="Target X coordinate")
    move_y: int = Field(description="Target Y coordinate")


class SugarscapeActionWithReasoning(BaseModel):
    """Structured action output with reasoning (for baseline profile).

    Attributes:
        reasoning: Explanation of why this move was chosen.
        move_x: X coordinate where the agent should move.
        move_y: Y coordinate where the agent should move.
    """

    reasoning: str = Field(description="Brief explanation of decision")
    move_x: int = Field(description="Target X coordinate")
    move_y: int = Field(description="Target Y coordinate")


class SugarscapeLLMStrategy(LLMStrategy):
    """LLM-driven strategy for Sugarscape agent movement.

    Uses structured prompts to guide an LLM in making movement decisions
    based on resource availability, agent state, and survival objectives.
    """

    def __init__(
        self,
        client: Any,
        verbose: bool = False,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize LLM strategy with configuration.

        Args:
            client: LLM client configured with instructor for structured outputs.
            verbose: If True, print LLM reasoning for each decision.
            config: LLMConfig object with optimization settings. Defaults to cached().
        """
        super().__init__(client)
        self.verbose = verbose
        self.config = config if config is not None else LLMConfig.cached()
        self.async_client: Any = None
        self.cache = DecisionCache() if self.config.use_cache else None

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for async decide method.

        Args:
            context: Contains position, vision, neighbors, agent resources, and grid.

        Returns:
            Action dictionary with 'move' key containing target position.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Use decide_async in async context")
        except RuntimeError:
            return asyncio.run(self.decide_async(context))

    async def decide_async(self, context: dict[str, Any]) -> dict[str, Any]:
        """Decide where to move using LLM inference.

        Args:
            context: Contains position, vision, neighbors, agent resources, and grid.

        Returns:
            Action dictionary with 'move' key containing target position.
        """
        current_pos = context["position"]
        sugar = context["sugar"]
        spice = context["spice"]
        metabolism_sugar = context["metabolism_sugar"]
        metabolism_spice = context["metabolism_spice"]
        neighbors = context["neighbors"]
        grid = context["grid"]

        neighbor_options = []
        for neighbor_cell in neighbors:
            neighbor_pos = neighbor_cell.coordinate
            is_occupied = len(neighbor_cell.agents) > 0

            sugar_at_pos = grid.sugar.data[neighbor_pos]
            spice_at_pos = grid.spice.data[neighbor_pos]

            neighbor_options.append(
                {
                    "position": neighbor_pos,
                    "sugar": int(sugar_at_pos),
                    "spice": int(spice_at_pos),
                    "occupied": is_occupied,
                    "distance": abs(current_pos[0] - neighbor_pos[0])
                    + abs(current_pos[1] - neighbor_pos[1]),
                }
            )

        current_sugar = grid.sugar.data[current_pos]
        current_spice = grid.spice.data[current_pos]
        neighbor_options.append(
            {
                "position": current_pos,
                "sugar": int(current_sugar),
                "spice": int(current_spice),
                "occupied": False,
                "distance": 0,
            }
        )

        if self.config.prompt_style == "long":
            system_prompt = (
                f"You are an agent in a Sugarscape simulation environment. "
                f"Your goal is to survive by collecting sugar and spice resources. "
                f"Each turn, you consume {metabolism_sugar} units of sugar and "
                f"{metabolism_spice} units of spice due to your metabolism. "
                f"If either resource reaches zero, you will die. "
                f"You should move to locations that maximize your resource collection "
                f"and ensure long-term survival."
            )

            pos_descriptions = []
            for opt in neighbor_options:
                if opt["occupied"]:
                    pos_descriptions.append(
                        f"Position {opt['position']}: OCCUPIED (cannot move here)"
                    )
                else:
                    pos_descriptions.append(
                        f"Position {opt['position']}: Sugar={opt['sugar']}, "
                        f"Spice={opt['spice']}, Distance={opt['distance']}"
                    )

            user_prompt = (
                f"Current Status:\n"
                f"- Your position: {current_pos}\n"
                f"- Your sugar: {sugar}\n"
                f"- Your spice: {spice}\n\n"
                f"Available moves:\n" + "\n".join(pos_descriptions) + "\n\n"
                "Choose the best position to move to and explain your reasoning."
            )
        else:
            system_prompt = (
                f"Sugarscape agent. Survive by collecting sugar+spice. "
                f"You use {metabolism_sugar}S+{metabolism_spice}Sp/turn. "
                f"Die if either reaches 0. Pick best move."
            )

            pos_list = []
            for opt in neighbor_options:
                if opt["occupied"]:
                    continue
                pos_list.append(f"{opt['position']}:S{opt['sugar']},Sp{opt['spice']}")

            user_prompt = (
                f"You:{current_pos} S{sugar},Sp{spice} Moves:{','.join(pos_list)}"
            )

        if self.config.use_cache and self.cache is not None:
            cached_decision = self.cache.get(sugar, spice, neighbor_options)
            if cached_decision is not None:
                valid_positions = [
                    opt["position"] for opt in neighbor_options if not opt["occupied"]
                ]
                if cached_decision in valid_positions:
                    if self.verbose:
                        agent_id = context.get("agent_id", "?")
                        print(
                            f"Agent {agent_id} at {current_pos} → {cached_decision} "
                            f"(cached)"
                        )
                    return {"move": cached_decision}

        response_model = (
            SugarscapeActionWithReasoning
            if self.config.include_reasoning
            else SugarscapeAction
        )

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

            chosen_pos = (response.move_x, response.move_y)

            valid_positions = [
                opt["position"] for opt in neighbor_options if not opt["occupied"]
            ]

            if chosen_pos not in valid_positions:
                chosen_pos = current_pos

            if self.config.use_cache and self.cache is not None:
                self.cache.put(sugar, spice, neighbor_options, chosen_pos)

            if self.verbose:
                agent_id = context.get("agent_id", "?")
                if self.config.include_reasoning and hasattr(response, "reasoning"):
                    print(
                        f"Agent {agent_id} at {current_pos} → {chosen_pos} "
                        f"(reasoning: {response.reasoning})"
                    )
                else:
                    print(f"Agent {agent_id} at {current_pos} → {chosen_pos}")

            return {"move": chosen_pos}

        except Exception as e:
            print(f"LLM call failed: {e}. Agent staying at current position.")
            return {"move": current_pos}

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics if caching is enabled.

        Returns:
            Dictionary with cache statistics or None if caching disabled.
        """
        if self.cache is not None:
            return self.cache.get_stats()
        return None
