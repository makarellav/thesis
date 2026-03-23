# Comparative Analysis: Algorithmic vs LLM-Based Agents in Multi-Agent Simulations

**Model**: gpt-4o-mini | **Temperatures**: 0.0, 0.3, 0.7 | **Seeds**: 456, 789, 1337 | **Profiles**: heuristic, optimized (LLM), optimized_with_reasoning (LLM + CoT)

---

## 1. Executive Summary

This analysis compares heuristic (rule-based) and LLM-driven agents across five classical agent-based models. The central finding is that **LLM suitability depends on whether individual rationality aligns with the collective emergent behavior** the simulation is designed to produce.

LLMs exhibit a consistent **self-preservation bias**: they avoid risk, conserve resources, and refuse actions that appear individually irrational — even when those actions are necessary for emergent dynamics. This bias manifests differently across simulations, creating a clear spectrum of LLM effectiveness:

| Rank | Simulation | LLM Fidelity | Core Issue |
|------|-----------|--------------|------------|
| 1 | Schelling Segregation | Perfect | Individual rationality = emergence mechanism |
| 2 | Sugarscape | High (with reasoning) | Reasoning enables near-heuristic survival |
| 3 | Wolf-Sheep Predation | Low | Conservation bias kills reproduction |
| 4 | Virus Spread (SIR) | Low | Social distancing suppresses epidemic dynamics |
| 5 | Boltzmann Wealth | Very Low | Rational self-interest prevents wealth transfer |

---

## 2. Per-Simulation Results

### 2.1 Schelling Segregation — LLM Success

**Outcome**: LLM agents perfectly reproduce the heuristic result.

| Metric | Heuristic | LLM (Optimized) | LLM (+ Reasoning) |
|--------|-----------|------------------|-------------------|
| % Happy | 100.0% | 100.0% | 100.0% |
| Segregation Index | 0.844 | 0.840 | 0.840 |
| Unhappy Agents | 0 | 0 | 0 |

**Figure reference**: `schelling_final_metrics.png` — all three bars are virtually identical across all metrics, with minimal error bars.

**Time-series** (`schelling_ts_segregation_index.png`): All three profiles follow nearly the same trajectory from ~0.55 to ~0.84 over 7 steps. The heuristic leads slightly in early steps but the LLM profiles converge to the same final value.

**Why it works**: The Schelling model's emergent behavior (segregation) arises from individually rational decisions — agents move when unhappy with their neighborhood composition. The LLM correctly interprets the threshold rule: "My similar_fraction is 0.29, below my threshold of 0.50, therefore I will move." This is a binary decision with clear numerical criteria, making it trivial for the LLM.

**Temperature effect**: None. All temperatures produce identical results (100% happy, ~0.84 segregation). The decision is binary and deterministic — temperature adds no meaningful variation.

**Decision analysis**: Only unhappy agents make decisions (happy=0.0 in all LLM logs). Agents converge rapidly — ~36 active agents in early steps dropping to ~1 by mid-simulation.

---

### 2.2 Sugarscape — Partial Success with Reasoning

**Outcome**: LLM with reasoning approaches heuristic survival; without reasoning, performance collapses.

| Metric (avg across seeds) | Heuristic | LLM (Optimized) | LLM (+ Reasoning) |
|---------------------------|-----------|------------------|-------------------|
| Survival Rate (temp 0.0) | 21.67% | 6.67% | 18.33% |
| Survival Rate (temp 0.3) | 21.67% | 13.33% | 23.33% |
| Survival Rate (temp 0.7) | 21.67% | 15.00% | 25.00% |
| Avg Wealth (temp 0.7) | 145.25 | 102.87 | 113.44 |
| Gini (temp 0.7) | 0.148 | 0.100 | 0.200 |

**Figure reference**: `sugarscape_final_metrics.png` (temp 0.0) — LLM Optimized shows drastically lower survival (orange bar ~1) vs heuristic (~4.3) and reasoning (~4). At temp 0.7, reasoning survival bar is comparable to heuristic.

**Time-series** (`sugarscape_ts_agent_count.png`): Heuristic (blue) maintains the highest agent count throughout. LLM + Reasoning (green) tracks closely but diverges slightly downward after step 20. LLM Optimized (orange) drops steeply and flatlines near zero by step 35.

**Key insight — reasoning is critical**: Chain-of-thought reasoning provides a ~3x improvement in survival over the optimized-only profile at temp 0.0 (18.33% vs 6.67%). Reasoning enables the LLM to articulate resource-seeking strategies rather than making blind movement decisions.

**Temperature effect**: Moderate and positive. Survival improves with temperature for both LLM profiles. At temp 0.7, reasoning agents (25%) actually exceed heuristic survival (21.67%), suggesting that some exploration helps in resource-scarce environments.

**Action distribution** (`sugarscape_actions_optimized_with_reasoning.png`): Move actions are well-distributed across the grid, indicating the LLM explores effectively rather than clustering. No dominant "stay in place" behavior, unlike other simulations.

---

### 2.3 Wolf-Sheep Predation — Conservation Bias Collapse

**Outcome**: Wolves go extinct in every LLM run. The predator-prey ecosystem collapses.

| Metric (avg across seeds) | Heuristic | LLM (Optimized) | LLM (+ Reasoning) |
|---------------------------|-----------|------------------|-------------------|
| Final Wolves (all temps) | 4.0 | 0.0 | 0.0 |
| Final Sheep (temp 0.0) | 3.3 | 6.0 | 5.3 |
| Total Population (temp 0.0) | 7.3 | 6.0 | 5.3 |
| Peak Wolves | 11.7 | 10.0 | 10.0 |

**Figure reference**: `wolfsheep_final_metrics.png` — Wolf Population shows heuristic at ~4 while both LLM bars are at exactly 0. Sheep population is actually higher for LLM (no predation pressure).

**Time-series** (`wolfsheep_ts_wolves.png`): Heuristic wolf population fluctuates between 3-12 over 50 steps (classic predator-prey oscillation). Both LLM profiles decline steadily from 10 to 0, reaching extinction by step 20-25. No recovery or oscillation occurs.

**Time-series** (`wolfsheep_ts_sheep.png`): Heuristic sheep drop sharply (predation) then recover around step 40 — classic Lotka-Volterra dynamics. LLM sheep decline from 20 to ~5 and flatline — no oscillation because wolves are gone.

**Action distribution** (`wolfsheep_actions_optimized_with_reasoning.png`): `reproduce=False` dominates (~1100 decisions), followed by `eat=False` (~800). The LLM consistently refuses to reproduce, citing "need to conserve energy for future actions."

**The conservation bias in detail**:
- Reasoning agents stay in place **40.8%** of the time (vs 0% heuristic)
- Average energy: 6.0-6.97 (LLM) vs 12.51 (heuristic) — agents are chronically energy-starved because they don't hunt effectively
- Wolves say "I will stay put to conserve energy" when no sheep are visible — but staying costs energy too, and they starve
- No agent ever reproduces — the LLM sees reproduction as an energy cost, never as a population necessity

**Why heuristic works**: The heuristic uses probabilistic reproduction (8% chance per step regardless of state). This "irrational" behavior — reproducing even when energy is moderate — maintains the population. The LLM's rational refusal to reproduce when energy is "not sufficient" creates a death spiral.

**Temperature effect**: Minimal. Wolves go extinct at all temperatures. Higher temp slightly reduces final sheep count (6.0 → 5.3 at temp 0.7 for reasoning) — more randomness doesn't help when the fundamental behavior is wrong.

---

### 2.4 Virus Spread (SIR) — Social Distancing Suppresses Dynamics

**Outcome**: LLM agents practice social distancing, reducing attack rates and suppressing the SIR epidemic curve.

| Metric (avg across seeds) | Heuristic | LLM (Optimized) | LLM (+ Reasoning) |
|---------------------------|-----------|------------------|-------------------|
| Attack Rate (all temps) | 41.7% | 25.0% | 25.0% |
| Final Resistant | 5.0 | 3.3 | 3.3 |
| Final Susceptible | 14.7 | 16.7 | 16.7 |
| Pct Infected (final) | 1.7% | 0.0% | 0.0% |

**Figure reference**: `virus_final_metrics.png` — Infected count shows heuristic with wide variance (0-0.8 range), while both LLM profiles are at exactly 0. Susceptible count is higher for LLM (more agents avoid infection). Resistant count is lower (fewer agents go through infection→recovery).

**Time-series** (`virus_ts_infected.png`): Heuristic (blue) shows the classic SIR curve — infections peak at ~8 around step 5, then gradually decline over 50 steps with high variance. LLM profiles (orange/green) decline monotonically from 5 with almost no variance and no secondary peaks. The LLM completely suppresses the infection spike that characterizes SIR dynamics.

**Movement analysis — the hermit effect**:

| Metric | Temp 0.0 | Temp 0.3 | Temp 0.7 | Heuristic |
|--------|----------|----------|----------|-----------|
| Stayed in place (Optimized) | 92.6% | 85.6% | 67.6% | 0.0% |
| Stayed in place (Reasoning) | 81.6% | 57.4% | 49.6% | 0.0% |
| Avg distance (Optimized) | 0.11 | 0.22 | 0.55 | 2.72 |
| Avg distance (Reasoning) | 0.35 | 0.80 | 0.95 | 2.72 |

**Action distribution** (`virus_actions_optimized_with_reasoning.png`): `interact=False` dominates with ~2900 out of 3000 decisions (~97%). The LLM refuses contact in almost every situation, even when the agent is already resistant.

**Reasoning reveals the logic**: Even resistant agents avoid infected neighbors — "To minimize exposure, I will move away from the infected neighbor." The LLM applies self-preservation even when the agent is immune, demonstrating that the bias is not context-dependent but systematic.

**Temperature effect**: This is the one simulation where temperature meaningfully changes behavior. At temp 0.7, optimized agents move from 92.6% stationary to 67.6%, and reasoning agents from 81.6% to 49.6%. However, the attack rate remains unchanged at 25.0% across all temperatures — the movement increase doesn't translate to more infections because agents still avoid interaction.

---

### 2.5 Boltzmann Wealth Model — Rational Self-Interest Prevents Emergence

**Outcome**: LLM agents refuse to trade, destroying the Boltzmann-Gibbs wealth distribution.

| Metric (avg across seeds) | Heuristic | LLM (Optimized) | LLM (+ Reasoning) |
|---------------------------|-----------|------------------|-------------------|
| Gini (temp 0.0) | 0.673 | 0.233 | 0.433 |
| Gini (temp 0.3) | 0.673 | 0.320 | 0.453 |
| Gini (temp 0.7) | 0.673 | 0.307 | 0.480 |
| Wealth Std Dev (temp 0.0) | 1.38 | 0.56 | 0.81 |
| Suboptimal Trade Rate | 81.4% | 100.0% | 71.4% |

**Figure reference**: `boltzmann_final_metrics.png` — Gini shows a dramatic gap: heuristic at ~0.67 (high inequality, as expected from random exchange), LLM Optimized at ~0.11 (near-equality), LLM Reasoning at ~0.43 (partial inequality). Wealth Std Dev shows the same pattern.

**Time-series** (`boltzmann_ts_gini_coefficient.png`): Heuristic Gini rises steadily to ~0.67 by step 25 and stabilizes — classic Boltzmann-Gibbs distribution forming. LLM Optimized (orange) barely rises above 0.1, essentially flat. LLM Reasoning (green) rises slowly to ~0.43 but never approaches heuristic levels.

**Action distribution** (`boltzmann_actions_optimized.png`): `give=False` accounts for ~1500 out of 1500 decisions. The LLM refuses to give wealth in virtually every case. With reasoning (`boltzmann_actions_optimized_with_reasoning.png`), `give=True` appears (~80 times) but `give=False` still dominates (~1400).

**The fundamental conflict**: The Boltzmann-Gibbs distribution emerges from random, unconditional exchange — agents give one unit of wealth to a random neighbor regardless of their own wealth. From an individual perspective, this is irrational (why give away wealth?). The LLM recognizes this irrationality and refuses, which is the "correct" individual decision but destroys the collective emergent pattern.

**Reasoning occasionally helps**: With CoT, agents sometimes justify trading: reasoning Gini (0.433-0.480) is significantly higher than optimized (0.233-0.320), indicating that the reasoning process can occasionally overcome the self-preservation default.

**Temperature effect**: Slight positive effect. Reasoning Gini improves from 0.433 to 0.480 with higher temperature. Higher randomness marginally increases the chance of giving.

---

## 3. Cross-Simulation Analysis

### 3.1 The Self-Preservation Bias

Across all five simulations, LLM agents demonstrate a consistent behavioral pattern:

| Behavior | Manifestation | Simulations Affected |
|----------|---------------|---------------------|
| **Risk avoidance** | Staying in place, avoiding contact | Virus (92.6% stationary), Wolf-Sheep (40.8%) |
| **Resource hoarding** | Refusing to give/share wealth | Boltzmann (100% refuse to trade) |
| **Reproduction avoidance** | Not reproducing to "conserve energy" | Wolf-Sheep (reproduce=False dominant) |
| **Interaction avoidance** | Refusing to interact with neighbors | Virus (97% interact=False) |

This bias is not a bug — it reflects the LLM's training on human text, where self-preservation and risk avoidance are generally rational. The problem is that emergent behaviors in ABMs often require individually "irrational" actions:
- Random wealth exchange (Boltzmann)
- Probabilistic reproduction regardless of energy state (Wolf-Sheep)
- Free movement without infection avoidance (Virus)

### 3.2 When Does Individual Rationality Align with Collective Emergence?

The key predictor of LLM success is whether the simulation's emergent behavior can be produced by individually rational agents:

**Alignment (LLM succeeds)**:
- **Schelling**: Agents seek similar neighbors → segregation emerges. The individual desire (comfort) directly produces the collective pattern (segregation). LLM perfectly replicates this.
- **Sugarscape**: Agents seek resources → survival. With reasoning, the LLM identifies resource-rich areas and navigates effectively. The individual goal (survive) aligns with the observable metric (survival rate).

**Misalignment (LLM fails)**:
- **Boltzmann**: The emergent distribution requires giving away wealth. No rational individual would do this voluntarily. LLM refusal rate: ~100%.
- **Virus**: The SIR dynamics require free mixing and contact. Rational agents avoid infection. LLM stationary rate: 67-93%.
- **Wolf-Sheep**: The ecosystem requires reproduction even at personal cost. Rational agents conserve energy. LLM reproduction rate: ~0%.

### 3.3 Effect of Chain-of-Thought Reasoning

Reasoning consistently improves LLM performance but cannot overcome fundamental misalignment:

| Simulation | Optimized | + Reasoning | Improvement |
|-----------|-----------|-------------|-------------|
| Sugarscape (survival, t=0.0) | 6.67% | 18.33% | **2.7x** |
| Boltzmann (Gini, t=0.0) | 0.233 | 0.433 | **1.9x** |
| Wolf-Sheep (final pop, t=0.0) | 6.0 | 5.3 | 0.9x (worse) |
| Virus (attack rate) | 25.0% | 25.0% | 1.0x (no change) |
| Schelling (segregation) | 0.840 | 0.840 | 1.0x (already perfect) |

Notable observations:
- **Sugarscape**: Reasoning provides the largest improvement — 3x better survival. The LLM can reason about resource locations and movement strategies.
- **Boltzmann**: Reasoning nearly doubles the Gini coefficient — the CoT process occasionally justifies trading ("I have more wealth, so I can share").
- **Wolf-Sheep**: Reasoning actually makes things slightly worse. Agents "overthink" and stay in place more (40.8% vs 2.4%), rationalizing inaction as "energy conservation."
- **Virus**: Reasoning has no effect on attack rate (25% for both) despite increasing mobility. Agents move more but still refuse to interact.

### 3.4 Effect of Temperature

| Simulation | Key Metric | Temp 0.0 | Temp 0.3 | Temp 0.7 | Sensitivity |
|-----------|------------|----------|----------|----------|-------------|
| Schelling | Segregation | 0.840 | 0.839 | 0.866 | None |
| Sugarscape | Survival (reasoning) | 18.33% | 23.33% | 25.00% | Moderate |
| Boltzmann | Gini (reasoning) | 0.433 | 0.453 | 0.480 | Low |
| Virus | Stayed in place (reasoning) | 81.6% | 57.4% | 49.6% | **High** |
| Wolf-Sheep | Final wolves | 0.0 | 0.0 | 0.0 | None |

Temperature is most effective for **continuous decisions with soft thresholds** (virus movement, sugarscape exploration) and least effective for **binary decisions with clear criteria** (Schelling move/stay, Boltzmann give/don't).

### 3.5 Computational Cost

| Simulation | Heuristic | LLM (Optimized) | LLM (+ Reasoning) | Slowdown |
|-----------|-----------|------------------|-------------------|----------|
| Schelling | 0.001s/step | 1.5s/step | 3.4s/step | 1500-3400x |
| Sugarscape | 0.001s/step | 1.9s/step | 3.0s/step | 1900-3000x |
| Boltzmann | 0.000s/step | 1.5s/step | 2.7s/step | ~1500-2700x |
| Virus | 0.001s/step | 1.7s/step | 2.5s/step | 1700-2500x |
| Wolf-Sheep | 0.001s/step | 3.1s/step | 4.5s/step | 3100-4500x |

LLM-based agents are **1500-4500x slower** than heuristic agents. This cost is only justified when:
1. The LLM produces equivalent emergent behavior (Schelling, Sugarscape with reasoning)
2. The research goal is to study the decision-making process itself (reasoning traces)
3. The simulation requires adaptive behavior that can't be easily coded as rules

### 3.6 Latency Distribution

| Simulation | Optimized (median) | Optimized (p95) | Reasoning (median) | Reasoning (p95) |
|-----------|-------------------|-----------------|--------------------|-----------------|
| Schelling | 960-1017ms | 1889-1921ms | 1836-2080ms | 2877-3204ms |
| Boltzmann | ~1000ms | ~1700ms | ~1700ms | ~2500ms |
| Virus | 1093-1184ms | 1468-2018ms | 1535-1793ms | 2078-2520ms |
| Wolf-Sheep | 1058-1133ms | 1973-2334ms | 1607-1749ms | 2740-3526ms |

Reasoning adds ~60-80% latency overhead vs optimized-only. P95 latencies are 1.5-2x the median, indicating occasional API slowdowns but generally stable performance.

---

## 4. Key Takeaways

### 4.1 LLMs as ABM Replacements

LLMs can replace algorithmic rules in agent-based models **only when the emergent behavior arises from individually rational decisions**. For simulations where emergence requires "irrational" actions (random exchange, unconditional reproduction, free mixing), LLMs systematically fail because their self-preservation bias overrides the behaviors necessary for collective patterns.

### 4.2 The Rationality-Emergence Paradox

Many classical ABM emergent behaviors depend on agents NOT being rational:
- The Boltzmann-Gibbs distribution requires random, unconditional wealth transfer
- Predator-prey oscillations require reproduction even at personal energy cost
- SIR dynamics require contact without infection avoidance

LLMs, trained on human reasoning, cannot easily produce these "irrational" behaviors. This is a fundamental limitation, not a prompt engineering problem.

### 4.3 Reasoning as a Partial Bridge

Chain-of-thought reasoning consistently improves performance by allowing the LLM to:
- Articulate strategies beyond simple self-preservation (Sugarscape: +3x survival)
- Occasionally justify altruistic actions (Boltzmann: +1.9x Gini)

However, reasoning can also backfire — in Wolf-Sheep, reasoning agents "overthink" and stay in place more, rationalizing inaction.

### 4.4 Temperature as a Weak Lever

Temperature helps most for continuous, non-binary decisions and is irrelevant for binary choices. Even at its most effective (Virus: mobility increase from 8% to 50%), it cannot overcome the fundamental behavioral bias — agents move more but still refuse to interact.

### 4.5 Practical Recommendations

1. **Use LLMs for Schelling-like simulations** where individual preferences drive collective patterns
2. **Use reasoning prompts** — they provide the largest consistent improvement
3. **Use moderate temperature (0.3-0.7)** for resource-gathering simulations
4. **Do not use LLMs** for simulations requiring probabilistic/random individual behavior (Boltzmann, SIR)
5. **Consider hybrid approaches** — mix heuristic agents (for "irrational" behaviors) with LLM agents (for rational decisions) to study how rational minorities affect emergent dynamics
