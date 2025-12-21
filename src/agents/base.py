"""Base agent implementation following the Strategy Pattern.

This module provides the foundational agent class that separates
agent embodiment (physical properties, state) from decision-making
logic (strategy implementation).
"""

from typing import Any

import mesa

from src.agents.strategy import DecisionStrategy


class BaseAgent(mesa.Agent):  # type: ignore[misc]
    """Base agent class implementing the Strategy Pattern.

    This class represents the "body" of an agent - its physical presence
    in the simulation, state tracking, and action execution. The "brain"
    (decision-making logic) is delegated to a DecisionStrategy instance.

    Attributes:
        unique_id: Unique identifier for this agent.
        model: Reference to the model instance containing this agent.
        strategy: DecisionStrategy instance handling decision-making.
    """

    def __init__(
        self,
        model: mesa.Model,
        strategy: DecisionStrategy,
    ) -> None:
        """Initialize the base agent.

        Args:
            model: The model instance this agent belongs to.
            strategy: The decision strategy this agent will use.
        """
        super().__init__(model)
        self.strategy = strategy

    def step(self) -> None:
        """Execute one step of agent behavior.

        This method delegates decision-making to the strategy, then
        executes the resulting action. Subclasses should override
        execute_action() to define simulation-specific behaviors.
        """
        context = self.get_context()
        action = self.strategy.decide(context)
        self.execute_action(action)

    def get_context(self) -> dict[str, Any]:
        """Gather environmental and agent state information.

        Returns:
            Dictionary containing context information needed for
            decision-making. Should be overridden by subclasses to
            provide simulation-specific context.
        """
        return {
            "agent_id": self.unique_id,
            "model": self.model,
        }

    def execute_action(self, action: dict[str, Any]) -> None:
        """Execute the decided action.

        Args:
            action: Action dictionary returned by the strategy's decide() method.

        Note:
            Should be overridden by subclasses to implement simulation-specific
            action execution logic.
        """
        pass
