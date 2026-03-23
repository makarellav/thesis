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

- Decisions: 373 | Errors: 0 | Steps: 50
- Latency: mean=1023.3ms, median=968.8ms, p95=1913.2ms

### Active Agents Over Time
  - early (0-16): avg 16.8 agents/step
  - mid (16-33): avg 1.5 agents/step
  - late (33-50): avg 1.0 agents/step

### Agent State at Decision Time
  - agent_type: mean=0.76, std=0.43, range=[0.0, 1.0]
  - homophily: mean=0.5, std=0.0, range=[0.5, 0.5]
  - similar_count: mean=1.64, std=1.0, range=[0.0, 3.0]
  - different_count: mean=4.61, std=1.15, range=[2.0, 8.0]
  - total_neighbors: mean=6.24, std=1.07, range=[3.0, 8.0]
  - similar_fraction: mean=0.26, std=0.15, range=[0.0, 0.43]
  - happy: mean=0.0, std=0.0, range=[0.0, 0.0]

---

## LLM (+ Reasoning)

- Decisions: 321 | Errors: 0 | Steps: 20
- Latency: mean=2155.9ms, median=1887.4ms, p95=3146.4ms

### Active Agents Over Time
  - early (0-6): avg 35.8 agents/step
  - mid (6-13): avg 5.3 agents/step
  - late (13-20): avg 1.0 agents/step

### Agent State at Decision Time
  - agent_type: mean=0.72, std=0.45, range=[0.0, 1.0]
  - homophily: mean=0.5, std=0.0, range=[0.5, 0.5]
  - similar_count: mean=1.82, std=0.94, range=[0.0, 3.0]
  - different_count: mean=4.48, std=1.09, range=[2.0, 8.0]
  - total_neighbors: mean=6.3, std=1.06, range=[3.0, 8.0]
  - similar_fraction: mean=0.29, std=0.14, range=[0.0, 0.43]
  - happy: mean=0.0, std=0.0, range=[0.0, 0.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 3, agent 76 (1)
  State: agent_type=1, homophily=0.5, similar_count=3, different_count=4, total_neighbors=7, similar_fraction=0.42857142857142855, happy=False
  Action: {'move': True}
  Reasoning: I have 3 similar neighbors and 4 different neighbors, which gives me a similarity fraction of 0.43. This is below my threshold of 0.50, meaning I am not satisfied with my current neighborhood composition. Therefore, I will choose to move to a new location to seek a more preferred neighborhood.

  **Sample 2** — step 4, agent 54 (1)
  State: agent_type=1, homophily=0.5, similar_count=1, different_count=3, total_neighbors=4, similar_fraction=0.25, happy=False
  Action: {'move': True}
  Reasoning: I have 1 similar neighbor and 3 different neighbors. This means that my current fraction of similar neighbors is 0.25 (1 similar out of 4 total neighbors). Since my threshold for moving is 50%, I do not meet the requirement to stay. Therefore, I will relocate to a new location to find a more favorable environment.

  **Sample 3** — step 11, agent 36 (1)
  State: agent_type=1, homophily=0.5, similar_count=2, different_count=5, total_neighbors=7, similar_fraction=0.2857142857142857, happy=False
  Action: {'move': True}
  Reasoning: I have 2 similar neighbors and 5 different neighbors, making a total of 7 neighbors. The fraction of similar neighbors is 2/7, which is approximately 0.29. Since this is below the threshold of 0.50, I feel uncomfortable and will relocate to find a more suitable environment.

  **Sample 4** — step 12, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=3, different_count=4, total_neighbors=7, similar_fraction=0.42857142857142855, happy=False
  Action: {'move': True}
  Reasoning: I currently have 3 similar neighbors out of a total of 7 neighbors. This means that only 43% of my neighbors are similar. My threshold is 50%, which means I need at least half of my neighbors to be similar in order to feel comfortable. Since I do not meet this threshold, I will relocate to find a more suitable environment.

  **Sample 5** — step 19, agent 18 (1)
  State: agent_type=1, homophily=0.5, similar_count=1, different_count=6, total_neighbors=7, similar_fraction=0.14285714285714285, happy=False
  Action: {'move': True}
  Reasoning: I have only 1 similar neighbor out of 7 total neighbors, which is approximately 14%. This is below my threshold of 50% for similar neighbors. Therefore, I will relocate to seek a more favorable environment.

---
