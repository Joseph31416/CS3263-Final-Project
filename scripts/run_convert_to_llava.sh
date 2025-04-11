#!/bin/bash

# Job name
#SBATCH --job-name=convert
#SBATCH --partition=cpu  

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=00:10:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 32 GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

## Log file names for output and error
#SBATCH --output=./logs/output_convert_%j.slurmlog
#SBATCH --error=./logs/error_convert_%j.slurmlog

# Default values
SRC_PATH=""

# Parse command-line arguments
while getopts "s:" opt; do
  case $opt in
    s) SRC_PATH="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Check if required arguments are provided
if [ -z "$SRC_PATH" ]; then
  echo "Error: Source path (-s) is required."
  exit 1
fi

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Remove cache
rm -r __pycache__

# Run Python script with arguments
python utils/convert_to_llava_text.py --src_path "$SRC_PATH" --num_cores "32"

# Sample Usage:
# sbatch scripts/run_convert_to_llava.sh -s ./data/text/test_puzzles.json
