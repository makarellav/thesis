# Schelling — Decision Log Summary

## Heuristic

- Decisions: 1947 | Errors: 0 | Steps: 10

### Active Agents Over Time
  - early (0-3): avg 84.0 agents/step
  - mid (3-6): avg 84.0 agents/step
  - late (6-10): avg 82.5 agents/step

### Agent State at Decision Time
  - agent_type: mean=0.36, std=0.48, range=[0.0, 1.0]
  - homophily: mean=0.5, std=0.0, range=[0.5, 0.5]
  - similar_count: mean=4.94, std=1.75, range=[0.0, 8.0]
  - different_count: mean=1.54, std=1.57, range=[0.0, 8.0]
  - total_neighbors: mean=6.48, std=1.05, range=[3.0, 8.0]
  - similar_fraction: mean=0.76, std=0.24, range=[0.0, 1.0]
  - happy: mean=0.89, std=0.31, range=[0.0, 1.0]

---

## LLM (Optimized)

- Decisions: 312 | Errors: 0 | Steps: 20
- Latency: mean=1045.7ms, median=960.0ms, p95=1889.3ms

### Active Agents Over Time
  - early (0-6): avg 35.7 agents/step
  - mid (6-13): avg 4.3 agents/step
  - late (13-20): avg 1.0 agents/step

### Agent State at Decision Time
  - agent_type: mean=0.71, std=0.45, range=[0.0, 1.0]
  - homophily: mean=0.5, std=0.0, range=[0.5, 0.5]
  - similar_count: mean=1.84, std=0.93, range=[0.0, 3.0]
  - different_count: mean=4.51, std=1.1, range=[2.0, 8.0]
  - total_neighbors: mean=6.34, std=1.06, range=[3.0, 8.0]
  - similar_fraction: mean=0.29, std=0.13, range=[0.0, 0.43]
  - happy: mean=0.0, std=0.0, range=[0.0, 0.0]

---

## LLM (+ Reasoning)

- Decisions: 314 | Errors: 0 | Steps: 20
- Latency: mean=1937.2ms, median=1836.1ms, p95=2877.1ms

### Active Agents Over Time
  - early (0-6): avg 35.8 agents/step
  - mid (6-13): avg 4.4 agents/step
  - late (13-20): avg 1.0 agents/step

### Agent State at Decision Time
  - agent_type: mean=0.72, std=0.45, range=[0.0, 1.0]
  - homophily: mean=0.5, std=0.0, range=[0.5, 0.5]
  - similar_count: mean=1.83, std=0.93, range=[0.0, 3.0]
  - different_count: mean=4.49, std=1.09, range=[2.0, 8.0]
  - total_neighbors: mean=6.32, std=1.06, range=[3.0, 8.0]
  - similar_fraction: mean=0.29, std=0.13, range=[0.0, 0.43]
  - happy: mean=0.0, std=0.0, range=[0.0, 0.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 0, agent 68 (1)
  State: agent_type=1, homophily=0.5, similar_count=2, different_count=5, total_neighbors=7, similar_fraction=0.2857142857142857, happy=False
  Action: {'move': True}
  Reasoning: I have 2 similar neighbors out of a total of 7 neighbors, which gives me a similarity fraction of approximately 0.29. This is below my threshold of 0.50, indicating that I am not satisfied with my current location. Therefore, I will relocate to find a more suitable neighborhood with more similar neighbors.

  **Sample 2** — step 6, agent 59 (0)
  State: agent_type=0, homophily=0.5, similar_count=3, different_count=5, total_neighbors=8, similar_fraction=0.375, happy=False
  Action: {'move': True}
  Reasoning: I have 3 similar neighbors out of a total of 8 neighbors, which gives me a similarity fraction of 0.38. This is below my threshold of 0.50, indicating that I am not sufficiently surrounded by similar neighbors. Therefore, I will move to a new location to seek a more favorable environment.

  **Sample 3** — step 9, agent 23 (1)
  State: agent_type=1, homophily=0.5, similar_count=2, different_count=5, total_neighbors=7, similar_fraction=0.2857142857142857, happy=False
  Action: {'move': True}
  Reasoning: I have 2 similar neighbors and 5 different neighbors, which gives me a fraction of similar neighbors of 0.29. This is below my threshold of 0.50, indicating that I am not satisfied with my current location. Therefore, I will relocate to find a more suitable environment.

  **Sample 4** — step 12, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=3, different_count=4, total_neighbors=7, similar_fraction=0.42857142857142855, happy=False
  Action: {'move': True}
  Reasoning: I have 3 similar neighbors and 4 different neighbors, making my similarity fraction 0.43. Since this is below my threshold of 0.50, I feel uncomfortable in my current location and will move to a new location.

  **Sample 5** — step 19, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=1, different_count=6, total_neighbors=7, similar_fraction=0.14285714285714285, happy=False
  Action: {'move': True}
  Reasoning: I have only 1 similar neighbor out of 7 total neighbors, which is approximately 14%. This is below my threshold of 50%, indicating that I am not in a satisfactory situation. Therefore, I will relocate to find a more suitable environment.

---
