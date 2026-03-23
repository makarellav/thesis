#!/bin/bash

MODEL="${1:-gpt-4o-mini}"

echo "========================================="
echo "Virus Spread Temperature Experiment Suite"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Temperatures: 0.0, 0.3, 0.7"
echo ""

TEMPS=("0.0" "0.3" "0.7")

for temp in "${TEMPS[@]}"; do
  echo ""
  echo "========================================="
  echo "TEMPERATURE: $temp"
  echo "========================================="
  echo ""

  bash run_virus_multiseed.sh "$MODEL" "$temp"

  echo ""
  echo "Cooling down for 30 seconds before next temperature..."
  sleep 30
done

echo ""
echo "========================================="
echo "All temperature experiments complete!"
echo "========================================="
echo ""
echo "Run compare_virus_results.sh to see comparison tables."
