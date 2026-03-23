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

- Decisions: 2999 | Errors: 1 | Steps: 50
- Latency: mean=1227.0ms, median=1102.6ms, p95=2017.8ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.11 cells (median: 0, max: 9)
  - Stayed in place: 92.6%
  - Unique cells targeted: 76

### Agent State at Decision Time
  - infected_neighbors: mean=0.04, std=0.22, range=[0.0, 2.0]
  - susceptible_neighbors: mean=0.8, std=1.33, range=[0.0, 7.0]
  - resistant_neighbors: mean=0.12, std=0.37, range=[0.0, 2.0]
  - total_neighbors: mean=0.96, std=1.45, range=[0.0, 7.0]
  - infection_risk: mean=0.03, std=0.15, range=[0.0, 1.0]
  - cellmate_infected: mean=0.01, std=0.11, range=[0.0, 1.0]
  - cellmate_count: mean=0.65, std=0.96, range=[0.0, 3.0]
  - infection_duration: mean=2.16, std=5.17, range=[0.0, 31.0]

### Errors
  Total: 1
  - (1x) InstructorRetryException: <failed_attempts>

<generation number="1">
<exception>
    Error code: 429

---

## LLM (+ Reasoning)

- Decisions: 2999 | Errors: 1 | Steps: 50
- Latency: mean=1605.1ms, median=1535.3ms, p95=2078.2ms

### Active Agents Over Time
  - early (0-16): avg 20.0 agents/step
  - mid (16-33): avg 20.0 agents/step
  - late (33-50): avg 20.0 agents/step

### Movement Behavior
  - Avg distance: 0.35 cells (median: 0, max: 9)
  - Stayed in place: 81.6%
  - Unique cells targeted: 96

### Agent State at Decision Time
  - infected_neighbors: mean=0.06, std=0.25, range=[0.0, 2.0]
  - susceptible_neighbors: mean=0.94, std=1.12, range=[0.0, 7.0]
  - resistant_neighbors: mean=0.09, std=0.32, range=[0.0, 3.0]
  - total_neighbors: mean=1.08, std=1.19, range=[0.0, 8.0]
  - infection_risk: mean=0.03, std=0.15, range=[0.0, 1.0]
  - cellmate_infected: mean=0.01, std=0.11, range=[0.0, 1.0]
  - cellmate_count: mean=0.43, std=0.74, range=[0.0, 3.0]
  - infection_duration: mean=2.16, std=5.17, range=[0.0, 31.0]

### Reasoning Samples (diverse)
  **Sample 1** — step 6, agent 5 (RESISTANT)
  State: state=RESISTANT, infected_neighbors=1, susceptible_neighbors=3, resistant_neighbors=1, total_neighbors=5, infection_risk=0.2, cellmate_infected=0, cellmate_count=0
  Action: {'move': [8, 0], 'interact': False}
  Reasoning: I am in a position with 1 infected neighbor, which poses a risk. To minimize exposure, I will move away from the infected neighbor while also considering the number of susceptible neighbors. Moving east (E) will take me away from the infected and towards more susceptible individuals, which is a safer option.

  **Sample 2** — step 15, agent 13 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=1, resistant_neighbors=0, total_neighbors=1, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [6, 3], 'interact': False}
  Reasoning: I am currently in a susceptible state and there is one susceptible neighbor. To minimize the risk of infection, I will move away from any potential infected areas. Since there are no infected individuals nearby, I will choose to stay in my current position and not interact.

  **Sample 3** — step 20, agent 5 (RESISTANT)
  State: state=RESISTANT, infected_neighbors=1, susceptible_neighbors=1, resistant_neighbors=0, total_neighbors=2, infection_risk=0.5, cellmate_infected=0, cellmate_count=0
  Action: {'move': [0, 4], 'interact': False}
  Reasoning: I am in a position with one infected neighbor, which poses a risk to my health. To minimize exposure, I will move away from the infected individual. Moving south (S) will take me to a position with a susceptible neighbor, which is safer than staying near the infected.

  **Sample 4** — step 38, agent 6 (SUSCEPTIBLE)
  State: state=SUSCEPTIBLE, infected_neighbors=0, susceptible_neighbors=4, resistant_neighbors=0, total_neighbors=4, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [2, 2], 'interact': False}
  Reasoning: I am currently in a susceptible state with no infected individuals nearby. To maintain my health, I will move away from potential areas of infection. I will move to the east (E) where I have more space and avoid any potential risks.

  **Sample 5** — step 49, agent 2 (RESISTANT)
  State: state=RESISTANT, infected_neighbors=0, susceptible_neighbors=2, resistant_neighbors=1, total_neighbors=3, infection_risk=0.0, cellmate_infected=0, cellmate_count=0
  Action: {'move': [3, 1], 'interact': True}
  Reasoning: I am in a position with no infected individuals nearby, and I have a couple of susceptible neighbors. I will move to the east to maintain distance from any potential threats while also being in a position to interact with the susceptible individuals if needed.

### Errors
  Total: 1
  - (1x) InstructorRetryException: <failed_attempts>

<generation number="1">
<exception>
    Error code: 429

---
