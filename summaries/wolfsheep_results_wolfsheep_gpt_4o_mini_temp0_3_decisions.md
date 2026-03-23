# Wolfsheep — Decision Log Summary

## Heuristic

- Decisions: 1694 | Errors: 0 | Steps: 50

### Active Agents Over Time
  - early (0-16): avg 29.4 agents/step
  - mid (16-33): avg 20.5 agents/step
  - late (33-50): avg 17.4 agents/step

### Movement Behavior
  - Avg distance: 1.94 cells (median: 1, max: 9)
  - Stayed in place: 0.0%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=12.51, std=12.8, range=[0.04, 81.46]
  - age: mean=12.82, std=11.87, range=[1.0, 50.0]
  - wolves_nearby_count: mean=0.18, std=0.43, range=[0.0, 3.0]
  - grass_available: mean=0.49, std=0.5, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.08, std=0.31, range=[0.0, 3.0]

---

## LLM (Optimized)

- Decisions: 1047 | Errors: 0 | Steps: 29
- Latency: mean=1294.4ms, median=1133.8ms, p95=2334.2ms

### Active Agents Over Time
  - early (0-9): avg 29.8 agents/step
  - mid (9-19): avg 21.9 agents/step
  - late (19-29): avg 8.6 agents/step

### Movement Behavior
  - Avg distance: 1.73 cells (median: 1, max: 9)
  - Stayed in place: 3.0%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=6.24, std=4.42, range=[0.0, 22.94]
  - age: mean=7.98, std=6.58, range=[0.0, 28.0]
  - wolves_nearby_count: mean=0.24, std=0.43, range=[0.0, 1.0]
  - grass_available: mean=0.41, std=0.49, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.09, std=0.3, range=[0.0, 2.0]

---

## LLM (+ Reasoning)

- Decisions: 1075 | Errors: 0 | Steps: 30
- Latency: mean=1987.9ms, median=1749.3ms, p95=3525.5ms

### Active Agents Over Time
  - early (0-10): avg 28.6 agents/step
  - mid (10-20): avg 22.8 agents/step
  - late (20-30): avg 15.5 agents/step

### Movement Behavior
  - Avg distance: 1.03 cells (median: 1, max: 9)
  - Stayed in place: 43.0%
  - Unique cells targeted: 97

### Agent State at Decision Time
  - energy: mean=6.97, std=5.01, range=[0.01, 25.06]
  - age: mean=9.59, std=7.92, range=[0.0, 29.0]
  - wolves_nearby_count: mean=0.09, std=0.29, range=[0.0, 2.0]
  - grass_available: mean=0.32, std=0.47, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.05, std=0.25, range=[0.0, 2.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 3, agent 102 (sheep)
  State: agent_type=sheep, energy=4.406434954052456, age=3, wolves_nearby_count=0, grass_available=True, sheep_in_cell=1
  Action: {'move': [5, 0], 'eat': True, 'reproduce': False}
  Reasoning: There is grass available to eat, and no wolves nearby, so I will eat the grass to gain energy. After eating, I will have enough energy to survive and will not reproduce this turn as I need to conserve energy for future actions.

  **Sample 2** — step 10, agent 121 (wolf)
  State: agent_type=wolf, energy=11.972033085803037, age=10, sheep_nearby_count=0, sheep_in_cell=0, wolves_in_cell=1
  Action: {'move': [2, 1], 'eat': False, 'reproduce': False}
  Reasoning: There are no sheep nearby to hunt, and I am currently alone with only one wolf in the vicinity. It is more strategic to stay put and conserve energy rather than move aimlessly without a target. I will also not eat or reproduce since there are no resources available.

  **Sample 3** — step 14, agent 130 (wolf)
  State: agent_type=wolf, energy=12.063455485919533, age=14, sheep_nearby_count=0, sheep_in_cell=0, wolves_in_cell=0
  Action: {'move': [1, 4], 'eat': False, 'reproduce': False}
  Reasoning: There are no sheep nearby or in my current location, so I will stay put to conserve energy. I will not eat or reproduce as there is no food available and I need to save energy for future actions.

  **Sample 4** — step 18, agent 110 (sheep)
  State: agent_type=sheep, energy=4.458379469036112, age=18, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [8, 7], 'eat': False, 'reproduce': False}
  Reasoning: There are no wolves nearby, and I have enough energy to move. I will move east to explore for grass and avoid any potential threats. I will not eat or reproduce at this time since there is no grass available.

  **Sample 5** — step 25, agent 107 (sheep)
  State: agent_type=sheep, energy=8.371778879555094, age=25, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [7, 7], 'eat': False, 'reproduce': False}
  Reasoning: There is no grass available to eat, and I need to move to find food. Since there are no wolves around, I can safely move in any direction. I will move East to explore for grass.

---
