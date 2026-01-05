#!/bin/bash

set -e

MODEL="${1:-gpt-4o-mini}"

echo "========================================"
echo "Temperature Experiment Suite"
echo "========================================"
echo ""
echo "Model: $MODEL"
echo "Profiles: heuristic, optimized, optimized_with_reasoning"
echo "Temperatures: 0.0, 0.1, 0.7"
echo "Seeds: 456, 789, 1337"
echo "Total runs: 27 (9 per temperature)"
echo ""
echo "Estimated time: 2-3 hours"
echo ""
read -p "Press Enter to start experiments..."
echo ""

echo ""
echo "========================================"
echo "Experiment 1/3: Temperature 0.0"
echo "========================================"
echo ""
./run_multiseed_comparison.sh "$MODEL" 0.0

echo ""
echo "Cooling down between experiments (30 seconds)..."
sleep 30

echo ""
echo "========================================"
echo "Experiment 2/3: Temperature 0.3"
echo "========================================"
echo ""
./run_multiseed_comparison.sh "$MODEL" 0.3

echo ""
echo "Cooling down between experiments (30 seconds)..."
sleep 30

echo ""
echo "========================================"
echo "Experiment 3/3: Temperature 0.7"
echo "========================================"
echo ""
./run_multiseed_comparison.sh "$MODEL" 0.7

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results directories:"
echo "  - results_multiseed_${MODEL}_temp0_0/"
echo "  - results_multiseed_${MODEL}_temp0_1/"
echo "  - results_multiseed_${MODEL}_temp0_7/"
echo ""
echo "Summary files:"
echo "  - results_multiseed_${MODEL}_temp0_0/summary.txt"
echo "  - results_multiseed_${MODEL}_temp0_1/summary.txt"
echo "  - results_multiseed_${MODEL}_temp0_7/summary.txt"
echo ""
echo "To compare results:"
echo "  cat results_multiseed_${MODEL}_temp*/summary.txt"
echo ""
