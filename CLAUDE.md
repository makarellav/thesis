# Bachelor's Thesis: Algorithmic vs LLM-Based Agents in Multi-Agent Simulations

## Project Overview

This thesis investigates whether Large Language Models (LLMs) can effectively replace hard-coded algorithmic rules in Agent-Based Models (ABM) while preserving emergent swarm behaviors. The research compares:

- **Algorithmic Agents**: Traditional rule-based agents with explicit if-then logic programmed directly into simulations
- **LLM-Driven Agents**: Prompt-based agents that receive environmental data as context and generate behaviors through natural language processing

The core research question examines whether LLMs can produce self-organizing patterns comparable to rule-based systems through prompt engineering, while potentially offering greater adaptability in hybrid configurations.

### Scientific Foundation

This work builds upon research into multi-agent simulations using localized decision-making entities that interact with their environment and other agents. We leverage the Mesa framework (Python) to implement classical ABM scenarios and extend them with LLM-based decision-making capabilities.

## Architecture Standards

### Strategy Pattern for Agent Decision-Making

**CRITICAL**: All agent implementations MUST follow the **Strategy Pattern** to decouple agent embodiment from decision-making logic.

#### Core Principle

- **Agent Body**: Represents the physical entity in the simulation (position, state, attributes)
- **Decision Strategy**: Encapsulates the "brain" or decision-making algorithm (algorithmic or LLM-based)

#### Implementation Requirements

```
Agent (Body)
├── DecisionStrategy (Interface/Abstract Base)
│   ├── AlgorithmicStrategy (Rule-based logic)
│   └── LLMStrategy (Prompt-based logic)
```

**Benefits:**
- Clean separation of concerns
- Easy switching between algorithmic and LLM decision-making
- Facilitates A/B testing and hybrid approaches
- Maintains single responsibility principle

**Example Pattern:**
```python
class Agent:
    def __init__(self, strategy: DecisionStrategy):
        self.strategy = strategy

    def step(self):
        action = self.strategy.decide(self.get_context())
        self.execute(action)
```

## Tech Stack

- **Language**: Python 3.12
- **ABM Framework**: Mesa 3.3.1+
- **Package Manager**: uv
- **Linting**: ruff (line-length 88, rules: E, F, I, N, W, UP, B, C4, SIM)
- **Type Checking**: mypy (strict mode, ignore_missing_imports)

## Commands

### Development Tools

- **Run Linter**: `uv run ruff check .`
- **Run Type Checker**: `uv run mypy .`
- **Run Simulation**: (TBD - will be added when simulations are implemented)

### Dependency Management

- **Install dependencies**: `uv sync`
- **Add dependency**: `uv add <package>`
- **Add dev dependency**: `uv add --dev <package>`

## Simulations Roadmap

The following classical ABM scenarios will be implemented with both algorithmic and LLM-based agent strategies:

1. **Sugarscape**
   - Resource gathering and wealth distribution
   - Agent movement, metabolism, vision

2. **Wolf-Sheep Predation**
   - Predator-prey dynamics
   - Energy transfer, reproduction, grass regrowth

3. **Schelling Segregation Model**
   - Spatial segregation emergence
   - Preference-based agent relocation

4. **Virus Spread**
   - Epidemiological dynamics
   - Infection, recovery, immunity states

5. **Boltzmann Wealth Model**
   - Economic exchange simulation
   - Wealth distribution emergence

Each simulation will be developed with:
- Algorithmic strategy implementation (baseline)
- LLM strategy implementation (experimental)
- Comparative analysis capabilities
- Visualization and data collection

## Project Structure

```
thesis/
├── src/
│   ├── agents/          # Agent base classes and strategy interfaces
│   ├── simulations/     # Individual simulation implementations
│   └── utils/           # Shared utilities, helpers, analysis tools
├── pyproject.toml       # Project configuration and dependencies
├── CLAUDE.md            # This file (source of truth)
└── README.md            # Project documentation
```

## Development Guidelines

1. **Type Hints**: All functions and methods must include type annotations
2. **Strategy Pattern**: Mandatory for all agent decision-making logic
3. **Documentation**: Clear docstrings for all public interfaces
4. **Testing**: (TBD - test framework to be added)
5. **Code Quality**: Pass ruff and mypy checks before commits

## Research Methodology

1. Implement baseline algorithmic version of each simulation
2. Develop LLM-based strategy variant for each scenario
3. Compare emergent behaviors through metrics:
   - Pattern formation
   - System stability
   - Computational efficiency
   - Behavioral diversity
4. Analyze hybrid approaches (mixed agent populations)

## Resources

### Research Paper
- [LLM-Driven Multi-Agent Systems Research](https://arxiv.org/html/2503.03800v1) - Core research on integrating LLMs into multi-agent simulations

### Mesa Framework
- [Mesa GitHub Repository](https://github.com/mesa/mesa) - Main ABM framework
- [Mesa Built-in Examples](https://github.com/mesa/mesa/tree/main/mesa/examples) - Framework-included example simulations
- [Mesa Documentation](https://github.com/mesa/mesa/tree/main/docs) - Official documentation

### Mesa LLM Integration
- [Mesa-LLM GitHub Repository](https://github.com/mesa/mesa-llm) - LLM integration extension for Mesa
- [Mesa-LLM Examples](https://github.com/mesa/mesa-llm/tree/main/examples) - LLM-based agent examples

### Community Examples
- [Mesa Examples Repository](https://github.com/mesa/mesa-examples) - Community-contributed simulation examples

## Notes

- Focus on reproducibility and clean architecture
- Document all design decisions and trade-offs
- Maintain clear separation between framework code and simulation-specific logic
- Prioritize code maintainability for future extensions
