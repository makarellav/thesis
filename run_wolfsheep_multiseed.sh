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
RESULTS_DIR="results_wolfsheep_${MODEL_SUFFIX}_temp${TEMP_SUFFIX}"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "Wolf-Sheep Predation - Multi-Seed Comparison"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperature: $TEMPERATURE"
echo "Results directory: $RESULTS_DIR"
echo "Testing ${#PROFILES[@]} profiles across ${#SEEDS[@]} seeds"
echo "Grid: 10x10, 20 sheep, 10 wolves, 50 steps"
echo "Total runs: $((${#PROFILES[@]} * ${#SEEDS[@]}))"
echo "Metrics: Wolves, Sheep, Grass, W/S Ratio, Total Population"
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
      uv run python -m src.simulations.run_wolf_sheep \
        --mode heuristic \
        --initial-sheep 20 \
        --initial-wolves 10 \
        --sheep-reproduce 0.08 \
        --wolf-reproduce 0.02 \
        --wolf-gain-from-food 10 \
        --grass-regrowth-time 15 \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/wolfsheep_output.txt | grep -E "Step [0-9]+|Simulation complete!|extinct"

      sed -n '/Simulation complete!/,$p' /tmp/wolfsheep_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f wolfsheep_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
    else
      uv run python -m src.simulations.run_wolf_sheep \
        --mode llm \
        --llm-profile "$profile" \
        --initial-sheep 20 \
        --initial-wolves 10 \
        --sheep-reproduce 0.08 \
        --wolf-reproduce 0.02 \
        --wolf-gain-from-food 10 \
        --grass-regrowth-time 15 \
        --width 10 \
        --height 10 \
        --seed "$seed" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --steps 50 \
        --log-decisions \
        --output-csv "$RESULTS_DIR/${profile}_seed${seed}.csv" \
        2>&1 | tee /tmp/wolfsheep_output.txt | grep -E "Step [0-9]+|Simulation complete!|extinct"

      sed -n '/Simulation complete!/,$p' /tmp/wolfsheep_output.txt > "$RESULTS_DIR/${profile}_seed${seed}.txt"
      mv -f wolfsheep_decisions_seed${seed}.json "$RESULTS_DIR/${profile}_seed${seed}_decisions.json" 2>/dev/null
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

echo "Wolf-Sheep Multi-Seed Comparison Summary" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for profile in "${PROFILES[@]}"; do
  echo "Profile: $profile" >> "$SUMMARY_FILE"
  echo "-------------------" >> "$SUMMARY_FILE"

  total_wolves=0
  total_sheep=0
  total_grass=0
  total_pop=0
  total_peak_wolves=0
  total_peak_sheep=0
  total_time=0
  count=0

  for seed in "${SEEDS[@]}"; do
    file="$RESULTS_DIR/${profile}_seed${seed}.txt"

    if [ -f "$file" ] && [ -s "$file" ]; then
      wolves=$(grep "Wolves:" "$file" | head -1 | grep -oP '\d+' || echo "0")
      sheep=$(grep "Sheep:" "$file" | head -1 | grep -oP '\d+' || echo "0")
      grass=$(grep "Grass:" "$file" | head -1 | grep -oP '\d+' || echo "0")
      total_population=$(grep "Total Population:" "$file" | head -1 | grep -oP '\d+' || echo "0")
      peak_wolves=$(grep "Peak Wolves:" "$file" | grep -oP '\d+' || echo "0")
      peak_sheep=$(grep "Peak Sheep:" "$file" | grep -oP '\d+' || echo "0")
      time=$(grep "Average time per step:" "$file" | grep -oP '\d+\.\d+' || echo "0.0")

      echo "  Seed $seed:" >> "$SUMMARY_FILE"
      echo "    Wolves: ${wolves}, Sheep: ${sheep}, Grass: ${grass}" >> "$SUMMARY_FILE"
      echo "    Total Pop: ${total_population}, Peak W: ${peak_wolves}, Peak S: ${peak_sheep}" >> "$SUMMARY_FILE"
      echo "    Time: ${time}s/step" >> "$SUMMARY_FILE"

      total_wolves=$(echo "$total_wolves + $wolves" | bc)
      total_sheep=$(echo "$total_sheep + $sheep" | bc)
      total_grass=$(echo "$total_grass + $grass" | bc)
      total_pop=$(echo "$total_pop + $total_population" | bc)
      total_peak_wolves=$(echo "$total_peak_wolves + $peak_wolves" | bc)
      total_peak_sheep=$(echo "$total_peak_sheep + $peak_sheep" | bc)
      total_time=$(echo "$total_time + $time" | bc)
      count=$((count + 1))
    fi
  done

  if [ $count -gt 0 ]; then
    avg_wolves=$(printf "%.1f" "$(echo "scale=3; $total_wolves / $count" | bc)")
    avg_sheep=$(printf "%.1f" "$(echo "scale=3; $total_sheep / $count" | bc)")
    avg_grass=$(printf "%.1f" "$(echo "scale=3; $total_grass / $count" | bc)")
    avg_pop=$(printf "%.1f" "$(echo "scale=3; $total_pop / $count" | bc)")
    avg_peak_wolves=$(printf "%.1f" "$(echo "scale=3; $total_peak_wolves / $count" | bc)")
    avg_peak_sheep=$(printf "%.1f" "$(echo "scale=3; $total_peak_sheep / $count" | bc)")
    avg_time=$(printf "%.3f" "$(echo "scale=5; $total_time / $count" | bc)")

    echo "" >> "$SUMMARY_FILE"
    echo "  AVERAGE:" >> "$SUMMARY_FILE"
    echo "    Wolves: ${avg_wolves}, Sheep: ${avg_sheep}, Grass: ${avg_grass}" >> "$SUMMARY_FILE"
    echo "    Total Pop: ${avg_pop}, Peak W: ${avg_peak_wolves}, Peak S: ${avg_peak_sheep}" >> "$SUMMARY_FILE"
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
