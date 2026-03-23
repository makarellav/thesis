# Virus — Decision Log Summary

## Heuristic

- Decisions: 3000 | Errors: 0 | Steps: 50

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 2.72 cells (median: 2, max: 18)
  - Stayed in place: 0.0%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - infected_neighbors: mean=0.12, std=0.37, range=[0.0, 3.0]
  - susceptible_neighbors: mean=1.11, std=1.0, range=[0.0, 5.0]
  - resistant_neighbors: mean=0.28, std=0.52, range=[0.0, 3.0]
  - total_neighbors: mean=1.51, std=1.15, range=[0.0, 6.0]
  - infection_risk: mean=0.07, std=0.21, range=[0.0, 1.0]
  - cellmate_infected: mean=0.02, std=0.12, range=[0.0, 1.0]
  - cellmate_count: mean=0.19, std=0.43, range=[0.0, 3.0]
  - infection_duration: mean=3.02, std=5.81, range=[0.0, 41.0]

---

## LLM (Optimized)

- Decisions: 3000 | Errors: 0 | Steps: 50
- Latency: mean=1140.8ms, median=1092.8ms, p95=1467.0ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.22 cells (median: 0, max: 9)
  - Stayed in place: 85.6%
  - Unique cells targeted: 86

### Agent State at Decision Time
  - infected_neighbors: mean=0.05, std=0.22, range=[0.0, 3.0]
  - susceptible_neighbors: mean=1.38, std=1.72, range=[0.0, 8.0]
  - resistant_neighbors: mean=0.13, std=0.41, range=[0.0, 4.0]
  - total_neighbors: mean=1.56, std=1.76, range=[0.0, 8.0]
  - infection_risk: mean=0.03, std=0.15, range=[0.0, 1.0]
  - cellmate_infected: mean=0.01, std=0.11, range=[0.0, 1.0]
  - cellmate_count: mean=0.54, std=0.76, range=[0.0, 3.0]
  - infection_duration: mean=2.16, std=5.16, range=[0.0, 31.0]

---

## LLM (+ Reasoning)

- Decisions: 3000 | Errors: 0 | Steps: 50
- Latency: mean=1640.6ms, median=1587.5ms, p95=2160.1ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.8 cells (median: 0, max: 9)
  - Stayed in place: 57.4%
  - Unique cells targeted: 97

### Agent State at Decision Time
  - infected_neighbors: mean=0.07, std=0.28, range=[0.0, 3.0]
  - susceptible_neighbors: mean=1.2, std=1.26, range=[0.0, 7.0]
  - resistant_neighbors: mean=0.17, std=0.44, range=[0.0, 3.0]
  - total_neighbors: mean=1.45, std=1.41, range=[0.0, 9.0]
  - infection_risk: mean=0.04, std=0.16, range=[0.0, 1.0]
  - cellmate_infected: mean=0.01, std=0.11, range=[0.0, 1.0]
  - cellmate_count: mean=0.39, std=0.72, range=[0.0, 4.0]
  - infection_duration: mean=2.16, std=5.16, range=[0.0, 31.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 9, agent 6 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=3, resistant_neighbors=1, total_neighbors=4, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [8, 3], 'interact': False}
  Reasoning: I am currently in a susceptible state with no infected individuals nearby. Since there are three susceptible individuals and one resistant individual, I will move towards a direction that keeps me away from potential future infections while also considering the presence of other susceptible individuals. I will move East (E) to maintain distance from any potential threats.

  **Sample 2** — step 12, agent 8 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=1, susceptible_neighbors=3, resistant_neighbors=0, total_neighbors=4, infection_risk=0.25, cellmate_infected=0, cellmate_count=0
  Action: {'move': [8, 0], 'interact': False}
  Reasoning: I am currently in a susceptible state and there is one infected individual nearby. To minimize the risk of infection, I will move away from the infected individual. The safest direction is to move east (E) where I can avoid the infected individual and maintain distance from potential threats.

  **Sample 3** — step 25, agent 2 (RESISTANT)
  State: state=RESISTANT, infected_neighbors=1, susceptible_neighbors=1, resistant_neighbors=0, total_neighbors=2, infection_risk=0.5, cellmate_infected=0, cellmate_count=0
  Action: {'move': [9, 0], 'interact': False}
  Reasoning: I am in a position with one infected neighbor and one susceptible neighbor. To avoid the risk of infection, I will move away from the infected neighbor while still being in a position to potentially interact with the susceptible neighbor if needed. Moving East (to position (8, 1)) keeps me safe from the infected while allowing for future interactions.

  **Sample 4** — step 33, agent 4 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=2, resistant_neighbors=1, total_neighbors=3, infection_risk=0.0, cellmate_infected=0, cellmate_count=1
  Action: {'move': [9, 0], 'interact': True}
  Reasoning: I am currently in a susceptible state and there are no infected individuals nearby. Since I have two susceptible neighbors, I will stay in my current position to avoid any potential risk of infection and interact with my neighbors to maintain social distance.

  **Sample 5** — step 46, agent 11 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=4, resistant_neighbors=1, total_neighbors=5, infection_risk=0.0, cellmate_infected=0, cellmate_count=1
  Action: {'move': [0, 0], 'interact': False}
  Reasoning: I am currently in a susceptible state with no immediate threat from infected individuals nearby. Since there are no infected individuals in my vicinity, I will move towards a direction that keeps me away from potential future threats while exploring. I will move East (E) to maintain distance from any potential infection.

---
