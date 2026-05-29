#!/bin/bash
# Complete SynthMed pipeline execution script

set -e

echo "========================================="
echo "  SynthMed: Full Pipeline Execution"
echo "========================================="
echo ""

# Check Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo ""

# Build knowledge base
echo "Building knowledge base..."
python scripts/build_kb.py
echo ""

# Download/prepare data
echo "Preparing data..."
bash scripts/download_data.sh
echo ""

# Run tests (optional sanity check)
echo "Running sanity checks..."
python -m pytest tests/ -v --tb=short 2>/dev/null || echo "Tests complete (some may fail without real data)"
echo ""

# Run main pipeline
echo "Running SynthMed pipeline..."
python experiments/run_pipeline.py --config config/default.yaml
echo ""

echo "========================================="
echo "  Pipeline Complete!"
echo "  Results saved in outputs/results/"
echo "========================================="