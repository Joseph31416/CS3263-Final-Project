#!/bin/bash

# Job name
#SBATCH --job-name=finetune
#SBATCH --partition=gpu-long

# GPU type constraint
#SBATCH --constraint=xgpi           # Use xgpi node
#SBATCH --gres=gpu:h100-47:1        # Use a single H100 GPU

#SBATCH --time=119:59:00            # Runtime duration
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --output=./logs/output_finetune_%j.slurmlog
#SBATCH --error=./logs/error_finetune_%j.slurmlog

# Default values
MODEL_ID=""
SRC_PATH=""
EVAL_PATH=""
NUM_EPOCHS=3
CHECKPOINT_PATH=""
LOGGING_STRATEGY="steps"
LOGGING_STEPS=250
EVAL_STEPS=500

# Parse long options
OPTS=$(getopt -o "" \
    --long model_id:,src_path:,eval_path:,num_epochs:,checkpoint_path:,logging_strategy:,logging_steps:,eval_steps: \
    -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
    echo "Failed parsing options." >&2
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        --model_id ) MODEL_ID="$2"; shift 2 ;;
        --src_path ) SRC_PATH="$2"; shift 2 ;;
        --eval_path ) EVAL_PATH="$2"; shift 2 ;;
        --num_epochs ) NUM_EPOCHS="$2"; shift 2 ;;
        --checkpoint_path ) CHECKPOINT_PATH="$2"; shift 2 ;;
        --logging_strategy ) LOGGING_STRATEGY="$2"; shift 2 ;;
        --logging_steps ) LOGGING_STEPS="$2"; shift 2 ;;
        --eval_steps ) EVAL_STEPS="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

# Required args check
if [ -z "$MODEL_ID" ]; then
    echo "Error: --model_id is required."
    exit 1
fi

if [ -z "$SRC_PATH" ]; then
    echo "Error: --src_path is required."
    exit 1
fi

if [ -z "$EVAL_PATH" ]; then
    echo "Error: --eval_path is required."
    exit 1
fi

# GPU status
nvidia-smi

# Activate environment
source ".venv/bin/activate"

# Clean Python cache
rm -r __pycache__ 2>/dev/null

# Run training script with all arguments
python finetuning/run_lora.py \
    --model_id "$MODEL_ID" \
    --src_path "$SRC_PATH" \
    --eval_path "$EVAL_PATH" \
    --num_epochs "$NUM_EPOCHS" \
    --logging_strategy "$LOGGING_STRATEGY" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}

# Supported model IDs:
# - "Qwen/Qwen2.5-VL-7B-Instruct"
# - "meta-llama/Llama-3.2-11B-Vision"
# - "microsoft/Phi-3.5-vision-instruct"
# - "Qwen/Qwen2.5-7B-Instruct"
# - "meta-llama/Meta-Llama-3.1-8B-Instruct"
# - "microsoft/Phi-3.5-mini-instruct"
