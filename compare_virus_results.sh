#!/bin/bash

MODEL="${1:-gpt-4o-mini}"
MODEL_SUFFIX=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')
OUTPUT_FILE="virus_comparison_${MODEL_SUFFIX}.txt"

{
echo "========================================="
echo "Virus Spread Temperature Comparison Analysis"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo ""

TEMPS=("0_0" "0_3" "0_7")
TEMP_LABELS=("0.0 (fully deterministic)" "0.3 (low exploration)" "0.7 (high exploration)")

for i in "${!TEMPS[@]}"; do
  temp="${TEMPS[$i]}"
  label="${TEMP_LABELS[$i]}"
  dir="results_virus_${MODEL_SUFFIX}_temp${temp}"

  echo "========================================="
  echo "Temperature ${label}"
  echo "========================================="

  if [ -f "$dir/summary.txt" ]; then
    cat "$dir/summary.txt"
  else
    echo "Results not found at: $dir/summary.txt"
  fi

  echo ""
done

get_avg() {
  local file="$1" profile="$2" metric="$3"
  if [ ! -f "$file" ]; then echo "N/A"; return; fi

  local block
  block=$(sed -n "/^Profile: ${profile}$/,/^Profile:/p" "$file" | head -n -1)

  local avg_block
  avg_block=$(echo "$block" | grep -A 3 "AVERAGE:")

  local val
  val=$(echo "$avg_block" | grep -oP "${metric}: \K[0-9.]+" || echo "N/A")
  echo "$val"
}

print_table() {
  local title="$1" metric="$2" suffix="$3"

  echo "========================================="
  echo "Quick Comparison Table - $title"
  echo "========================================="
  echo ""
  printf "%-30s | %-12s | %-12s | %-12s\n" "Profile" "Temp 0.0" "Temp 0.3" "Temp 0.7"
  echo "----------------------------------------------------------------------"

  for profile in "heuristic" "optimized" "optimized_with_reasoning"; do
    row="$profile"

    for temp in "${TEMPS[@]}"; do
      dir="results_virus_${MODEL_SUFFIX}_temp${temp}"
      val=$(get_avg "$dir/summary.txt" "$profile" "$metric")
      row="$row | ${val}${suffix}"
    done

    printf "%-30s %s\n" "$row"
  done
  echo ""
}

print_table "Pct Infected (Final)" "Pct Infected" "%"
print_table "Attack Rate" "Attack Rate" "%"
print_table "Resistant (Final)" "Resistant" ""
print_table "Time per Step" "Time" "s/step"
} | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
