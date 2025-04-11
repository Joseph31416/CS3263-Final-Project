#!/bin/bash
# Job name
#SBATCH --job-name=install_venv
#SBATCH --partition=cpu  

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=01:00:00  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 20 GB of memory
#SBATCH --ntasks=1                          # Number of tasks (1 task)
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=32G                           # 20GB of memory per task

## Log file names for output and error
#SBATCH --output=./logs/output_instal_venv_cpu_%j.slurmlog
#SBATCH --error=./logs/error_instal_venv_cpu_%j.slurmlog

nvidia-smi

source ".venv/bin/activate"

rm -r __pycache__
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
