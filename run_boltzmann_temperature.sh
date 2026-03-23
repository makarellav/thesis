#!/bin/bash

set -e

MODEL="${1:-gpt-4o-mini}"

echo "========================================="
echo "Boltzmann Temperature Experiment Suite"
echo "========================================="
echo ""
echo "Model: $MODEL"
echo "Profiles: heuristic, optimized, optimized_with_reasoning"
echo "Temperatures: 0.0, 0.3, 0.7"
echo "Seeds: 456, 789, 1337"
echo "Total runs: 27 (9 per temperature)"
echo ""
echo "Metrics: Gini, Avg Wealth, Wealth Std Dev, Suboptimal Trade Rate"
echo ""
read -p "Press Enter to start experiments..."
echo ""

echo ""
echo "========================================="
echo "Experiment 1/3: Temperature 0.0"
echo "========================================="
echo ""
./run_boltzmann_multiseed.sh "$MODEL" 0.0

echo ""
echo "Cooling down between experiments (30 seconds)..."
sleep 30

echo ""
echo "========================================="
echo "Experiment 2/3: Temperature 0.3"
echo "========================================="
echo ""
./run_boltzmann_multiseed.sh "$MODEL" 0.3

echo ""
echo "Cooling down between experiments (30 seconds)..."
sleep 30

echo ""
echo "========================================="
echo "Experiment 3/3: Temperature 0.7"
echo "========================================="
echo ""
./run_boltzmann_multiseed.sh "$MODEL" 0.7

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================="
echo ""
MODEL_SUFFIX=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9]/_/g')

echo "Results directories:"
echo "  - results_boltzmann_${MODEL_SUFFIX}_temp0_0/"
echo "  - results_boltzmann_${MODEL_SUFFIX}_temp0_3/"
echo "  - results_boltzmann_${MODEL_SUFFIX}_temp0_7/"
echo ""
echo "To compare results across temperatures:"
echo "  ./compare_boltzmann_results.sh $MODEL"
echo ""
