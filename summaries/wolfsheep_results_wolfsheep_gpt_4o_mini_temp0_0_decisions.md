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

- Decisions: 1052 | Errors: 0 | Steps: 29
- Latency: mean=1125.8ms, median=1058.7ms, p95=1992.5ms

### Active Agents Over Time
  - early (0-9): avg 30.3 agents/step
  - mid (9-19): avg 21.6 agents/step
  - late (19-29): avg 7.9 agents/step

### Movement Behavior
  - Avg distance: 1.73 cells (median: 1, max: 9)
  - Stayed in place: 2.4%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=6.36, std=4.52, range=[0.0, 23.94]
  - age: mean=7.98, std=6.5, range=[0.0, 28.0]
  - wolves_nearby_count: mean=0.19, std=0.39, range=[0.0, 1.0]
  - grass_available: mean=0.41, std=0.49, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.06, std=0.25, range=[0.0, 1.0]

---

## LLM (+ Reasoning)

- Decisions: 1109 | Errors: 0 | Steps: 28
- Latency: mean=1747.3ms, median=1643.2ms, p95=2739.6ms

### Active Agents Over Time
  - early (0-9): avg 29.3 agents/step
  - mid (9-18): avg 22.4 agents/step
  - late (18-28): avg 12.4 agents/step

### Movement Behavior
  - Avg distance: 1.02 cells (median: 1, max: 9)
  - Stayed in place: 40.8%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=6.0, std=4.41, range=[0.0, 23.18]
  - age: mean=8.51, std=6.86, range=[0.0, 27.0]
  - wolves_nearby_count: mean=0.09, std=0.29, range=[0.0, 2.0]
  - grass_available: mean=0.31, std=0.46, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.08, std=0.3, range=[0.0, 2.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 1, agent 108 (sheep)
  State: agent_type=sheep, energy=4.134502146078539, age=1, wolves_nearby_count=1, grass_available=False, sheep_in_cell=1
  Action: {'move': [4, 7], 'eat': False, 'reproduce': False}
  Reasoning: There is one wolf nearby, which poses a threat. I will move away from the wolf to avoid danger. Since there is no grass to eat, I will not eat. I will also not reproduce as I need to conserve energy and focus on survival.

  **Sample 2** — step 9, agent 101 (sheep)
  State: agent_type=sheep, energy=5.036931509236837, age=9, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [1, 9], 'eat': False, 'reproduce': False}
  Reasoning: There is no grass available to eat, and I need to move to find food. Since there are no wolves present, I can safely move in any direction. I will move East to explore for grass or other sheep.

  **Sample 3** — step 12, agent 105 (sheep)
  State: agent_type=sheep, energy=6.052019783452094, age=12, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [9, 4], 'eat': False, 'reproduce': False}
  Reasoning: There is no grass available to eat, and no wolves or sheep nearby. I will stay in my current position to conserve energy and wait for potential food sources or other sheep to appear.

  **Sample 4** — step 20, agent 103 (sheep)
  State: agent_type=sheep, energy=18.13045613696336, age=20, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [2, 3], 'eat': False, 'reproduce': False}
  Reasoning: There is no grass available to eat, and no other sheep or wolves nearby. I will stay in my current position to conserve energy and wait for grass to grow or for other sheep to appear.

  **Sample 5** — step 27, agent 125 (wolf)
  State: agent_type=wolf, energy=0.17850016415401626, age=27, sheep_nearby_count=1, sheep_in_cell=0, wolves_in_cell=0
  Action: {'move': [6, 2], 'eat': False, 'reproduce': False}
  Reasoning: There is one sheep nearby, and I need to move towards it to hunt. Since I have no energy, I cannot eat or reproduce. I will move towards the sheep to increase my chances of hunting it next turn.

---
