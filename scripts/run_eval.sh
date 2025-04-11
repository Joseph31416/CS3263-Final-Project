#!/bin/bash

# Job name
#SBATCH --job-name=eval
#SBATCH --partition=gpu

## GPU type constraint: A100-40 on xgph node or H100-96 on xgpi node
## #SBATCH --constraint=xgph # Use A100-40 GPU
#SBATCH --constraint=xgpi # Use H100-96 GPU

## Request the appropriate GPU:
## #SBATCH --gres=gpu:a100-40:1  # Use A100-40 GPU
#SBATCH --gres=gpu:h100-47:1  # Use H100-96 GPU

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=4:59:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 32 GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

## Log file names for output and error
#SBATCH --output=./logs/output_eval_%j.slurmlog
#SBATCH --error=./logs/error_eval_%j.slurmlog

# Default values
ADAPTER_PATH=""
EVAL_PATH=""
EVAL_SCRIPT=""

# Parse command-line arguments
while getopts "s:a:e:" opt; do
  case $opt in
    s) EVAL_SCRIPT="$OPTARG" ;;
    a) ADAPTER_PATH="$OPTARG" ;;
    e) EVAL_PATH="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Check if required arguments are provided
if [ -z "$ADAPTER_PATH" ]; then
  echo "Error: Adapter path (-a) is required."
  exit 1
fi

if [ -z "$EVAL_PATH" ]; then
  echo "Error: Evaluation path (-e) is required."
  exit 1
fi

if [ -z "$EVAL_SCRIPT" ]; then
  echo "Error: Evaluation script (-s) is required."
  exit 1
fi

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Remove cache
rm -r __pycache__

# Run Python script with arguments
python "$EVAL_SCRIPT" --adapter_path "$ADAPTER_PATH" --eval_path "$EVAL_PATH"

# Sample Usage:
# sbatch scripts/run_eval.sh -s evaluation/evaluate_qwen_vl.py -a ./lora_output/train_puzzles_llava/training_args/checkpoint-500 -e ./data/text/test_puzzles_llava.json
