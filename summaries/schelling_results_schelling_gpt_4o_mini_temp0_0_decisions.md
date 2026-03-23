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

- Decisions: 314 | Errors: 0 | Steps: 20
- Latency: mean=1125.2ms, median=1016.7ms, p95=1921.5ms

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

---

## LLM (+ Reasoning)

- Decisions: 314 | Errors: 0 | Steps: 20
- Latency: mean=2215.0ms, median=2079.5ms, p95=3203.9ms

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
  **Sample 1** — step 0, agent 75 (0)
  State: agent_type=0, homophily=0.5, similar_count=2, different_count=5, total_neighbors=7, similar_fraction=0.2857142857142857, happy=False
  Action: {'move': True}
  Reasoning: I have 2 similar neighbors out of a total of 7 neighbors, which gives me a similarity fraction of approximately 0.29. This is below my threshold of 0.50, indicating that I am not satisfied with my current neighborhood composition. Therefore, I will move to a new location.

  **Sample 2** — step 4, agent 67 (1)
  State: agent_type=1, homophily=0.5, similar_count=3, different_count=5, total_neighbors=8, similar_fraction=0.375, happy=False
  Action: {'move': True}
  Reasoning: I have 3 similar neighbors and 5 different neighbors, which means my fraction of similar neighbors is 0.375 (3 out of 8). This is below my threshold of 0.50, indicating that I am not satisfied with my current location. Therefore, I will move to a new location.

  **Sample 3** — step 11, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=1, different_count=5, total_neighbors=6, similar_fraction=0.16666666666666666, happy=False
  Action: {'move': True}
  Reasoning: I have 1 similar neighbor and 5 different neighbors, which means my fraction of similar neighbors is 1/(1+5) = 1/6 ≈ 0.17. This is below the threshold of 0.50, indicating that I am not satisfied with my current location. Therefore, I will move to a new location to find a more favorable environment.

  **Sample 4** — step 12, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=3, different_count=4, total_neighbors=7, similar_fraction=0.42857142857142855, happy=False
  Action: {'move': True}
  Reasoning: I have 3 similar neighbors and 4 different neighbors, which gives me a similarity fraction of 0.43. Since this is below my threshold of 0.50, I feel uncomfortable in my current location and will move to find a more suitable environment.

  **Sample 5** — step 16, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=0, different_count=6, total_neighbors=6, similar_fraction=0.0, happy=False
  Action: {'move': True}
  Reasoning: I have no similar neighbors and a high number of different neighbors. Since my threshold is 50% and I am currently at 0% similar neighbors, I will move to a new location to find a more suitable environment.

---
