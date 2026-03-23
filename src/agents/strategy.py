from abc import ABC, abstractmethod
from typing import Any


class DecisionStrategy(ABC):

    @abstractmethod
    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        pass


class HeuristicStrategy(DecisionStrategy):

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"action": "idle"}


class LLMStrategy(DecisionStrategy):

    def __init__(self, client: Any) -> None:
        self.client = client

    def decide(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"action": "idle"}
