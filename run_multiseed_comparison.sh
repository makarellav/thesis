#!/bin/bash

PROFILES=(
  "heuristic"
  "optimized"
  "optimized_with_reasoning"
)

SEEDS=(1337)

MODEL="${1:-gpt-4o-mini}"
TEMPERATURE="${2:-0.1}"

MODEL_SUFFIX=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
TEMP_SUFFIX=$(echo "$TEMPERATURE" | sed 's/\./_/g')
RESULTS_DIR="results_multiseed_${MODEL_SUFFIX}_temp${TEMP_SUFFIX}"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Multi-Seed Profile Comparison"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "Testing ${#PROFILES[@]} profiles across ${#SEEDS[@]} seeds (20 agents each)"
echo "Total runs: $((${#PROFILES[@]} * ${#SEEDS[@]}))"
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
      uv run python -m src.simulations.run_sugarscape \
        --mode heuristic \
        --num-agents 20 \
        --seed "$seed" \
        2>&1 | tee /tmp/run_output.txt | grep -E "Step [0-9]+\.\.\.|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/run_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
    else
      uv run python -m src.simulations.run_sugarscape \
        --mode llm \
        --llm-profile "$profile" \
        --num-agents 20 \
        --seed "$seed" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        2>&1 | tee /tmp/run_output.txt | grep -E "Step [0-9]+\.\.\.|Simulation complete!"

      sed -n '/Simulation complete!/,$p' /tmp/run_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
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

echo "Multi-Seed Comparison Summary" >> "$SUMMARY_FILE"
echo "=============================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for profile in "${PROFILES[@]}"; do
  echo "Profile: $profile" >> "$SUMMARY_FILE"
  echo "-------------------" >> "$SUMMARY_FILE"

  total_survival=0
  total_time=0
  count=0

  for seed in "${SEEDS[@]}"; do
    file="$RESULTS_DIR/${profile}_seed${seed}.txt"

    if [ -f "$file" ]; then
      survival=$(grep "Survival Rate:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")
      time=$(grep "Average time per step:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")

      echo "  Seed $seed: ${survival}% survival, ${time}s/step" >> "$SUMMARY_FILE"

      total_survival=$(echo "$total_survival + $survival" | bc)
      total_time=$(echo "$total_time + $time" | bc)
      count=$((count + 1))
    fi
  done

  if [ $count -gt 0 ]; then
    avg_survival=$(echo "scale=2; $total_survival / $count" | bc)
    avg_time=$(echo "scale=3; $total_time / $count" | bc)

    echo "" >> "$SUMMARY_FILE"
    echo "  AVERAGE: ${avg_survival}% survival, ${avg_time}s/step" >> "$SUMMARY_FILE"
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
