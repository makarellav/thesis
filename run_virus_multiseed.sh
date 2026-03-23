#!/bin/bash

PROFILES=(
  "heuristic"
  "optimized"
  "optimized_with_reasoning"
)

SEEDS=(456 789 1337)

MODEL="${1:-gpt-4o-mini}"
TEMPERATURE="${2:-0.1}"

MODEL_SUFFIX=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
TEMP_SUFFIX=$(echo "$TEMPERATURE" | sed 's/\./_/g')
RESULTS_DIR="results_virus_${MODEL_SUFFIX}_temp${TEMP_SUFFIX}"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Virus Spread (SIR) - Multi-Seed Comparison"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "Testing ${#PROFILES[@]} profiles across ${#SEEDS[@]} seeds"
echo "Grid: 10x10, 20 agents, 5 initially infected, 50 steps"
echo "Total runs: $((${#PROFILES[@]} * ${#SEEDS[@]}))"
echo "Metrics: Infected, Susceptible, Resistant, Pct Infected, Attack Rate"
echo ""

for seed in "${SEEDS[@]}"; do
  echo "========================================"
  echo "SEED: $seed"
  echo "========================================"

  for profile in "${PROFILES[@]}"; do
    echo ""
    echo ">>> Running: $profile (seed=$seed)..."
    echo ""

    if [ "$profile" = "heuristic" ]; then
      uv run python -m src.simulations.run_virus \
        --mode heuristic \
        --num-agents 20 \
        --initial-infected 5 \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/virus_output.txt | grep -E "Step [0-9]+|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/virus_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f virus_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    else
      uv run python -m src.simulations.run_virus \
        --mode llm \
        --llm-profile "$profile" \
        --num-agents 20 \
        --initial-infected 5 \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/virus_output.txt | grep -E "Step [0-9]+|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/virus_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f virus_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    fi

    echo ""
    echo "Completed: $profile (seed=$seed)"
    echo "Cooling down for 10 seconds..."
    sleep 10
  done
  echo ""
done

echo "========================================="
echo "Aggregating Results..."
echo "========================================="
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
> "$SUMMARY_FILE"

echo "Virus Spread Multi-Seed Comparison Summary" >> "$SUMMARY_FILE"
echo "===========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for profile in "${PROFILES[@]}"; do
  echo "Profile: $profile" >> "$SUMMARY_FILE"
  echo "-------------------" >> "$SUMMARY_FILE"

  total_infected=0
  total_susceptible=0
  total_resistant=0
  total_pct_infected=0
  total_attack_rate=0
  total_time=0
  count=0

  for seed in "${SEEDS[@]}"; do
    file="$RESULTS_DIR/${profile}_seed${seed}.txt"

    if [ -f "$file" ] && [ -s "$file" ]; then
      infected=$(grep "Infected:" "$file" | grep -oP '\d+' | head -1 || echo "0")
      susceptible=$(grep "Susceptible:" "$file" | grep -oP '\d+' | head -1 || echo "0")
      resistant=$(grep "Resistant:" "$file" | grep -oP '\d+' | head -1 || echo "0")
      pct_infected=$(grep "Pct Infected:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      attack_rate=$(grep "Attack Rate:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      time=$(grep "Average time per step:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")

      echo "  Seed $seed:" >> "$SUMMARY_FILE"
      echo "    Infected: ${infected}, Susceptible: ${susceptible}, Resistant: ${resistant}" >> "$SUMMARY_FILE"
      echo "    Pct Infected: ${pct_infected}%, Attack Rate: ${attack_rate}%" >> "$SUMMARY_FILE"
      echo "    Time: ${time}s/step" >> "$SUMMARY_FILE"

      total_infected=$(echo "$total_infected + $infected" | bc)
      total_susceptible=$(echo "$total_susceptible + $susceptible" | bc)
      total_resistant=$(echo "$total_resistant + $resistant" | bc)
      total_pct_infected=$(echo "$total_pct_infected + $pct_infected" | bc)
      total_attack_rate=$(echo "$total_attack_rate + $attack_rate" | bc)
      total_time=$(echo "$total_time + $time" | bc)
      count=$((count + 1))
    fi
  done

  if [ $count -gt 0 ]; then
    avg_infected=$(printf "%.1f" "$(echo "scale=3; $total_infected / $count" | bc)")
    avg_susceptible=$(printf "%.1f" "$(echo "scale=3; $total_susceptible / $count" | bc)")
    avg_resistant=$(printf "%.1f" "$(echo "scale=3; $total_resistant / $count" | bc)")
    avg_pct_infected=$(printf "%.1f" "$(echo "scale=3; $total_pct_infected / $count" | bc)")
    avg_attack_rate=$(printf "%.1f" "$(echo "scale=3; $total_attack_rate / $count" | bc)")
    avg_time=$(printf "%.3f" "$(echo "scale=5; $total_time / $count" | bc)")

    echo "" >> "$SUMMARY_FILE"
    echo "  AVERAGE:" >> "$SUMMARY_FILE"
    echo "    Infected: ${avg_infected}, Susceptible: ${avg_susceptible}, Resistant: ${avg_resistant}" >> "$SUMMARY_FILE"
    echo "    Pct Infected: ${avg_pct_infected}%, Attack Rate: ${avg_attack_rate}%" >> "$SUMMARY_FILE"
    echo "    Time: ${avg_time}s/step" >> "$SUMMARY_FILE"
  fi

  echo "" >> "$SUMMARY_FILE"
done

echo "========================================="
echo "Complete! Results saved to:"
echo "  - Individual runs: $RESULTS_DIR/*_seed*.txt"
echo "  - Summary: $SUMMARY_FILE"
echo "========================================="
echo ""

cat "$SUMMARY_FILE"
