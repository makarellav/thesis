"""Demo script to run the Sugarscape simulation.

This script demonstrates the Sugarscape model in action with
heuristic-based agents. It initializes the model, creates agents
with strategies, runs the simulation, and displays results.
"""

from src.simulations.sugarscape import SugarAgent, SugarscapeModel
from src.simulations.sugarscape_strategy import SugarscapeHeuristicStrategy


def main() -> None:
    """Run a Sugarscape simulation demo."""
    print("=" * 60)
    print("Sugarscape Simulation Demo")
    print("=" * 60)

    model = SugarscapeModel(
        width=50,
        height=50,
        initial_population=100,
    )

    print(f"\nInitializing {100} agents with heuristic strategies...")

    for _i in range(100):
        sugar = model.random.randint(25, 50)
        spice = model.random.randint(25, 50)
        metabolism_sugar = model.random.randint(1, 5)
        metabolism_spice = model.random.randint(1, 5)
        vision = model.random.randint(1, 5)

        x = model.random.randrange(model.width)
        y = model.random.randrange(model.height)
        cell = model.grid[(x, y)]

        strategy = SugarscapeHeuristicStrategy()
        SugarAgent(
            model=model,
            strategy=strategy,
            cell=cell,
            sugar=sugar,
            spice=spice,
            metabolism_sugar=metabolism_sugar,
            metabolism_spice=metabolism_spice,
            vision=vision,
        )

    print(f"✓ Created {len(model.agents)} agents")

    num_steps = 50
    print(f"\nRunning simulation for {num_steps} steps...")
    print("-" * 60)
    print(f"{'Step':<8} {'Agents':<12} {'Avg Sugar':<12} {'Avg Spice':<12}")
    print("-" * 60)

    for step in range(num_steps):
        model.step()

        agent_count = len(model.agents)
        if agent_count > 0:
            sugar_agents = [a for a in model.agents if isinstance(a, SugarAgent)]
            avg_sugar = sum(a.sugar for a in sugar_agents) / len(sugar_agents)
            avg_spice = sum(a.spice for a in sugar_agents) / len(sugar_agents)
        else:
            avg_sugar = 0
            avg_spice = 0

        if step % 5 == 0 or agent_count == 0:
            print(f"{step:<8} {agent_count:<12} {avg_sugar:<12.2f} {avg_spice:<12.2f}")

        if agent_count == 0:
            print("\n⚠ All agents have starved!")
            break

    print("-" * 60)
    print("\nSimulation complete!")
    print(f"Final agent count: {len(model.agents)}")

    if len(model.datacollector.model_vars["Agent Count"]) > 0:
        print("\nAgent Population Over Time:")
        data = model.datacollector.model_vars["Agent Count"]
        print(f"  Initial: {data[0]}")
        print(f"  Final: {data[-1]}")
        print(f"  Survival Rate: {data[-1] / data[0] * 100:.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
