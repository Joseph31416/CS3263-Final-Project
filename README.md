# CS3263-Final-Project

## Setup Instructions

### 1. Create a Virtual Environment
Run the following command to create a virtual environment:
```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment
- On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```
- On Windows:
    ```bash
    .venv\Scripts\activate
    ```

### 3. Install Requirements
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

On the SoC cluster, you should use the given scripts to install the required packages via slurm:
```bash
sbatch scripts/install_venv_gpu.sh
sbatch scripts/install_venv_cpu.sh
```

## Usage Instructions

### Dataset Conversion
Create the image dataset from the given dataset. For example, to create the dataset from the `dataset/checkmate_in_one/` folder, run:
```bash
sbatch scripts/run_convert_to_llava.sh -s ./dataset/checkmate_in_one/train.json
```
Note that you have to perform this conversion for both the training and validation datasets.

### Training
Once the dataset is created, you can run the training script:
```bash
sbatch scripts/run_finetuning.sh --model_id <model_id> -s ./data/text/<task>/train.json -e ./data/text/<task>/test.json -n 3
```
Refer to the [`scripts/run_finetuning.sh`](scripts/run_finetuning.sh) file for more options.

### Evaluation
To evaluate the model, you can run the evaluation script:
```bash
sbatch scripts/run_eval.sh -s evaluation/evaluate_qwen_vl.py -a ./lora_output/<model_id>/<training_session>/checkpoints/<checkpoint> -e ./data/text/<task>/test.json
```

### List of models:
- "Qwen/Qwen2.5-VL-7B-Instruct"
- "meta-llama/Llama-3.2-11B-Vision"
- "microsoft/Phi-3.5-vision-instruct"
- "Qwen/Qwen2.5-7B-Instruct"
- "meta-llama/Meta-Llama-3.1-8B-Instruct"
- "microsoft/Phi-3.5-mini-instruct"

### List of tasks:
- "annotation_matching"
- "board_state_tracking"
- "checkmate_in_one"
- "position_evaluation"
- "puzzle_solving"
- "puzzle_solving_small"
- "puzzle_solving_large"
- "ensemble"

## Acknowledgements

The finetuning scripts are based off:
- [Qwen2-Vision-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)
- [Phi3-Vision-Finetune](https://github.com/2U1/Phi3-Vision-Finetune)
- [Llama3-Vision-Finetune](https://github.com/2U1/Llama3.2-Vision-Finetune)

The dataset is sourced from:
- [ChessGPT](https://github.com/waterhorse1/ChessGPT)
- [Big-Bench](https://github.com/google/BIG-bench)