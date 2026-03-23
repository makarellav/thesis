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
RESULTS_DIR="results_boltzmann_${MODEL_SUFFIX}_temp${TEMP_SUFFIX}"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Boltzmann Wealth Model - Multi-Seed Comparison"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "Testing ${#PROFILES[@]} profiles across ${#SEEDS[@]} seeds (10 agents each)"
echo "Total runs: $((${#PROFILES[@]} * ${#SEEDS[@]}))"
echo "Metrics: Gini, Avg Wealth, Wealth Std Dev, Suboptimal Trade Rate"
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
      uv run python -m src.simulations.run_boltzmann \
        --mode heuristic \
        --num-agents 10 \
        --seed "$seed" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/boltzmann_output.txt | grep -E "Step [0-9]+\.\.\.|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/boltzmann_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f boltzmann_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    else
      uv run python -m src.simulations.run_boltzmann \
        --mode llm \
        --llm-profile "$profile" \
        --num-agents 10 \
        --seed "$seed" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/boltzmann_output.txt | grep -E "Step [0-9]+\.\.\.|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/boltzmann_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f boltzmann_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    fi

    echo ""
    echo "✓ Completed: $profile (seed=$seed)"
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

echo "Boltzmann Multi-Seed Comparison Summary" >> "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for profile in "${PROFILES[@]}"; do
  echo "Profile: $profile" >> "$SUMMARY_FILE"
  echo "-------------------" >> "$SUMMARY_FILE"

  total_gini=0
  total_wealth=0
  total_std=0
  total_suboptimal=0
  total_time=0
  count=0

  for seed in "${SEEDS[@]}"; do
    file="$RESULTS_DIR/${profile}_seed${seed}.txt"

    if [ -f "$file" ]; then
      gini=$(grep "Gini Coefficient:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      wealth=$(grep "Average Wealth:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      std=$(grep "Wealth Std Dev:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      suboptimal=$(grep "Suboptimal Trade Rate:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      time=$(grep "Average time per step:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")

      echo "  Seed $seed:" >> "$SUMMARY_FILE"
      echo "    Gini: ${gini}, Avg Wealth: ${wealth}, Std Dev: ${std}" >> "$SUMMARY_FILE"
      echo "    Suboptimal: ${suboptimal}%, Time: ${time}s/step" >> "$SUMMARY_FILE"

      total_gini=$(echo "$total_gini + $gini" | bc)
      total_wealth=$(echo "$total_wealth + $wealth" | bc)
      total_std=$(echo "$total_std + $std" | bc)
      total_suboptimal=$(echo "$total_suboptimal + $suboptimal" | bc)
      total_time=$(echo "$total_time + $time" | bc)
      count=$((count + 1))
    fi
  done

  if [ $count -gt 0 ]; then
    avg_gini=$(printf "%.3f" "$(echo "scale=5; $total_gini / $count" | bc)")
    avg_wealth=$(printf "%.2f" "$(echo "scale=4; $total_wealth / $count" | bc)")
    avg_std=$(printf "%.2f" "$(echo "scale=4; $total_std / $count" | bc)")
    avg_suboptimal=$(printf "%.2f" "$(echo "scale=4; $total_suboptimal / $count" | bc)")
    avg_time=$(printf "%.3f" "$(echo "scale=5; $total_time / $count" | bc)")

    echo "" >> "$SUMMARY_FILE"
    echo "  AVERAGE:" >> "$SUMMARY_FILE"
    echo "    Gini: ${avg_gini}, Avg Wealth: ${avg_wealth}, Std Dev: ${avg_std}" >> "$SUMMARY_FILE"
    echo "    Suboptimal: ${avg_suboptimal}%, Time: ${avg_time}s/step" >> "$SUMMARY_FILE"
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
