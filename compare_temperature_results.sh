#!/bin/bash

MODEL="${1:-gpt-4o-mini}"
MODEL_SUFFIX=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')

echo "========================================"
echo "Temperature Comparison Analysis"
echo "========================================"
echo ""
echo "Model: $MODEL"
echo ""

TEMPS=("0_0" "0_1" "0_7")
TEMP_LABELS=("0.0 (fully deterministic)" "0.1 (low exploration)" "0.7 (high exploration)")

for i in "${!TEMPS[@]}"; do
  temp="${TEMPS[$i]}"
  label="${TEMP_LABELS[$i]}"
  dir="results_multiseed_${MODEL_SUFFIX}_temp${temp}"

  echo "========================================"
  echo "Temperature ${label}"
  echo "========================================"

  if [ -f "$dir/summary.txt" ]; then
    cat "$dir/summary.txt"
  else
    echo "⚠ Results not found at: $dir/summary.txt"
  fi

  echo ""
done

echo "========================================"
echo "Quick Comparison Table"
echo "========================================"
echo ""
printf "%-30s | %-15s | %-15s | %-15s\n" "Profile" "Temp 0.0" "Temp 0.1" "Temp 0.7"
echo "------------------------------------------------------------------------"

for profile in "heuristic" "optimized" "optimized_with_reasoning"; do
  row="$profile"

  for temp in "${TEMPS[@]}"; do
    dir="results_multiseed_${MODEL_SUFFIX}_temp${temp}"
    summary_file="$dir/summary.txt"

    if [ -f "$summary_file" ]; then
      avg=$(grep -A 10 "Profile: $profile" "$summary_file" | grep "AVERAGE:" | grep -oP '\d+\.\d+(?=% survival)' || echo "N/A")
      row="$row | $avg%"
    else
      row="$row | N/A"
    fi
  done

  printf "%-30s %s\n" "$row"
done
