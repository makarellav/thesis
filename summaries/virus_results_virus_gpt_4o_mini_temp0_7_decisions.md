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
- Latency: mean=1255.2ms, median=1183.5ms, p95=1688.0ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.55 cells (median: 0, max: 9)
  - Stayed in place: 67.6%
  - Unique cells targeted: 99

### Agent State at Decision Time
  - infected_neighbors: mean=0.06, std=0.25, range=[0.0, 2.0]
  - susceptible_neighbors: mean=1.18, std=1.21, range=[0.0, 7.0]
  - resistant_neighbors: mean=0.17, std=0.39, range=[0.0, 2.0]
  - total_neighbors: mean=1.41, std=1.28, range=[0.0, 7.0]
  - infection_risk: mean=0.03, std=0.16, range=[0.0, 1.0]
  - cellmate_infected: mean=0.02, std=0.12, range=[0.0, 1.0]
  - cellmate_count: mean=0.33, std=0.56, range=[0.0, 3.0]
  - infection_duration: mean=2.16, std=5.16, range=[0.0, 31.0]

---

## LLM (+ Reasoning)

- Decisions: 3000 | Errors: 0 | Steps: 50
- Latency: mean=1871.4ms, median=1793.8ms, p95=2519.8ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.95 cells (median: 1, max: 10)
  - Stayed in place: 49.6%
  - Unique cells targeted: 100

### Agent State at Decision Time
  - infected_neighbors: mean=0.08, std=0.29, range=[0.0, 3.0]
  - susceptible_neighbors: mean=1.33, std=1.21, range=[0.0, 6.0]
  - resistant_neighbors: mean=0.24, std=0.53, range=[0.0, 3.0]
  - total_neighbors: mean=1.65, std=1.41, range=[0.0, 9.0]
  - infection_risk: mean=0.04, std=0.15, range=[0.0, 1.0]
  - cellmate_infected: mean=0.01, std=0.1, range=[0.0, 1.0]
  - cellmate_count: mean=0.25, std=0.5, range=[0.0, 2.0]
  - infection_duration: mean=2.16, std=5.16, range=[0.0, 31.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 6, agent 11 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=2, resistant_neighbors=0, total_neighbors=2, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [5, 3], 'interact': False}
  Reasoning: I am currently in a safe position with no infected individuals nearby. To maintain my health, I will stay in my current position and avoid unnecessary movement. This will minimize the risk of encountering any potential infected individuals. Additionally, there are no infected agents in my vicinity, so I will not interact with anyone.

  **Sample 2** — step 12, agent 5 (RESISTANT)
  State: state=RESISTANT, infected_neighbors=1, susceptible_neighbors=2, resistant_neighbors=0, total_neighbors=3, infection_risk=0.3333333333333333, cellmate_infected=0, cellmate_count=1
  Action: {'move': [7, 1], 'interact': False}
  Reasoning: I am in a position with one infected neighbor at (6, 1) and two susceptible neighbors at (7, 0) and (7, 2). To minimize the risk of infection, I should stay put and not interact with the infected neighbor while keeping an eye on the susceptible ones. Staying allows me to avoid potential contact with the infected individual, maintaining my resistant state.

  **Sample 3** — step 28, agent 7 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=0, resistant_neighbors=0, total_neighbors=0, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [7, 7], 'interact': False}
  Reasoning: I am currently in a susceptible state with no infected individuals nearby. To stay safe, I will move in a direction that avoids potential risks while also maintaining the ability to interact if necessary. Since there are no infected individuals, I will choose to move East, which provides a safer path forward.

  **Sample 4** — step 30, agent 17 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=1, susceptible_neighbors=2, resistant_neighbors=1, total_neighbors=4, infection_risk=0.25, cellmate_infected=0, cellmate_count=1
  Action: {'move': [0, 0], 'interact': False}
  Reasoning: There is one infected individual nearby and two susceptible individuals. Given the risk level is low (0.25) and there are more susceptible individuals around, I should move away from the infected individual to reduce the chance of infection. Moving towards the east (E) keeps me away from the infected while still allowing me to stay close to other susceptible individuals.

  **Sample 5** — step 41, agent 17 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=2, resistant_neighbors=3, total_neighbors=5, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [3, 0], 'interact': False}
  Reasoning: Since I am susceptible and there are no infected individuals nearby, I will move to a neighboring position that has a higher count of susceptible individuals. Moving East (to position (3, 0)) seems appropriate as it is safe and may provide better interaction opportunities with other susceptible individuals.

---
