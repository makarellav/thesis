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

- Decisions: 1066 | Errors: 0 | Steps: 29
- Latency: mean=1106.9ms, median=1060.0ms, p95=1973.3ms

### Active Agents Over Time
  - early (0-9): avg 30.3 agents/step
  - mid (9-19): avg 22.8 agents/step
  - late (19-29): avg 8.0 agents/step

### Movement Behavior
  - Avg distance: 1.68 cells (median: 1, max: 9)
  - Stayed in place: 4.3%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=6.25, std=4.43, range=[0.0, 24.46]
  - age: mean=8.05, std=6.48, range=[0.0, 28.0]
  - wolves_nearby_count: mean=0.17, std=0.38, range=[0.0, 1.0]
  - grass_available: mean=0.41, std=0.49, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.06, std=0.25, range=[0.0, 2.0]

---

## LLM (+ Reasoning)

- Decisions: 1078 | Errors: 0 | Steps: 30
- Latency: mean=1780.7ms, median=1607.4ms, p95=2844.7ms

### Active Agents Over Time
  - early (0-10): avg 29.6 agents/step
  - mid (10-20): avg 21.5 agents/step
  - late (20-30): avg 10.2 agents/step

### Movement Behavior
  - Avg distance: 1.06 cells (median: 1, max: 9)
  - Stayed in place: 40.4%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - energy: mean=6.64, std=4.65, range=[0.0, 27.77]
  - age: mean=8.77, std=7.25, range=[0.0, 29.0]
  - wolves_nearby_count: mean=0.1, std=0.31, range=[0.0, 2.0]
  - grass_available: mean=0.31, std=0.46, range=[0.0, 1.0]
  - sheep_in_cell: mean=0.06, std=0.24, range=[0.0, 2.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 0, agent 115 (sheep)
  State: agent_type=sheep, energy=1.568234682263919, age=0, wolves_nearby_count=1, grass_available=False, sheep_in_cell=0
  Action: {'move': [3, 1], 'eat': False, 'reproduce': False}
  Reasoning: There is a wolf nearby, which poses a threat. I will move away from the wolf to avoid danger and search for grass to eat. Since there is no grass available, I won't eat, and I will not reproduce due to my low energy. Moving East is the best option to create distance from the wolf.

  **Sample 2** — step 6, agent 111 (sheep)
  State: agent_type=sheep, energy=3.2960170479681095, age=6, wolves_nearby_count=0, grass_available=False, sheep_in_cell=1
  Action: {'move': [7, 9], 'eat': False, 'reproduce': False}
  Reasoning: There are no wolves in the vicinity, which allows me to move safely. I will move East to explore new territory and look for grass to eat. Since no grass is available, I will not eat, but I will consider reproducing to increase my population if conditions are favorable next turn.

  **Sample 3** — step 16, agent 125 (wolf)
  State: agent_type=wolf, energy=0.46026132872688663, age=16, sheep_nearby_count=1, sheep_in_cell=0, wolves_in_cell=1
  Action: {'move': [2, 7], 'eat': False, 'reproduce': False}
  Reasoning: There is one sheep nearby, which presents an opportunity to hunt. However, there is also one wolf present, which could pose a threat. Since my energy is currently at 0, I need to move towards the sheep to potentially eat it and gain energy. I will move East to pursue the sheep.

  **Sample 4** — step 22, agent 103 (sheep)
  State: agent_type=sheep, energy=4.508372574431453, age=22, wolves_nearby_count=0, grass_available=True, sheep_in_cell=0
  Action: {'move': [2, 2], 'eat': True, 'reproduce': False}
  Reasoning: There is grass available, which is the food source for the prey agent. I will eat the grass to gain energy and survive. There are no wolves in the vicinity, so I can safely eat without the risk of being attacked. After eating, I will stay in the same position as I have enough energy and there is no immediate need to move or reproduce.

  **Sample 5** — step 28, agent 102 (sheep)
  State: agent_type=sheep, energy=11.158358394742486, age=28, wolves_nearby_count=0, grass_available=False, sheep_in_cell=0
  Action: {'move': [5, 4], 'eat': False, 'reproduce': False}
  Reasoning: There are no wolves present, so I can safely move in search of grass. Since there is no grass available, I will not eat. I will also not reproduce as I have no energy to spare for that. My best move is to go east to explore further for resources.

---
