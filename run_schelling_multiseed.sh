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
RESULTS_DIR="results_schelling_${MODEL_SUFFIX}_temp${TEMP_SUFFIX}"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Schelling Segregation - Multi-Seed Comparison"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "Testing ${#PROFILES[@]} profiles across ${#SEEDS[@]} seeds"
echo "Grid: 10x10, density 0.8, homophily 0.5"
echo "Total runs: $((${#PROFILES[@]} * ${#SEEDS[@]}))"
echo "Metrics: Pct Happy, Segregation Index, Num Unhappy"
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
      uv run python -m src.simulations.run_schelling \
        --mode heuristic \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/schelling_output.txt | grep -E "Step [0-9]+|Equilibrium|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/schelling_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f schelling_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    else
      uv run python -m src.simulations.run_schelling \
        --mode llm \
        --llm-profile "$profile" \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/schelling_output.txt | grep -E "Step [0-9]+|Equilibrium|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/schelling_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f schelling_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
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

echo "Schelling Multi-Seed Comparison Summary" >> "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for profile in "${PROFILES[@]}"; do
  echo "Profile: $profile" >> "$SUMMARY_FILE"
  echo "-------------------" >> "$SUMMARY_FILE"

  total_pct_happy=0
  total_segregation=0
  total_unhappy=0
  total_time=0
  count=0

  for seed in "${SEEDS[@]}"; do
    file="$RESULTS_DIR/${profile}_seed${seed}.txt"

    if [ -f "$file" ]; then
      pct_happy=$(grep "Pct Happy:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      segregation=$(grep "Segregation Index:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      unhappy=$(grep "Num Unhappy:" "$file" | grep -oP '\d+' || echo "0")
      time=$(grep "Average time per step:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")

      echo "  Seed $seed:" >> "$SUMMARY_FILE"
      echo "    Pct Happy: ${pct_happy}%, Segregation: ${segregation}, Unhappy: ${unhappy}" >> "$SUMMARY_FILE"
      echo "    Time: ${time}s/step" >> "$SUMMARY_FILE"

      total_pct_happy=$(echo "$total_pct_happy + $pct_happy" | bc)
      total_segregation=$(echo "$total_segregation + $segregation" | bc)
      total_unhappy=$(echo "$total_unhappy + $unhappy" | bc)
      total_time=$(echo "$total_time + $time" | bc)
      count=$((count + 1))
    fi
  done

  if [ $count -gt 0 ]; then
    avg_pct_happy=$(printf "%.1f" "$(echo "scale=3; $total_pct_happy / $count" | bc)")
    avg_segregation=$(printf "%.3f" "$(echo "scale=5; $total_segregation / $count" | bc)")
    avg_unhappy=$(printf "%.1f" "$(echo "scale=3; $total_unhappy / $count" | bc)")
    avg_time=$(printf "%.3f" "$(echo "scale=5; $total_time / $count" | bc)")

    echo "" >> "$SUMMARY_FILE"
    echo "  AVERAGE:" >> "$SUMMARY_FILE"
    echo "    Pct Happy: ${avg_pct_happy}%, Segregation: ${avg_segregation}, Unhappy: ${avg_unhappy}" >> "$SUMMARY_FILE"
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
