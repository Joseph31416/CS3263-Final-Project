import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import argparse
import json
from tqdm import tqdm
from evaluator import CombinedEvaluator
from PIL import Image
from env.env import LLAMA_API_KEY

def load_dataset(path: str) -> list[dict]:
    """
    Load the dataset from the given path.
    """
    with open(path, "r") as f:
        data = json.load(f)
    input_data = []
    labels = []
    images = []
    for row in data:
        image = row["image"]
        images.append(image)
        conversations = row["conversations"]
        human_conv = conversations[0]["value"]
        human_conv = human_conv.lstrip("<image>").strip()
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{human_conv}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        input_data.append(prompt)
        labels.append(conversations[1]["value"])
    return input_data, labels, images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter")
    parser.add_argument("--eval_path", type=str, help="Path to evaluation data")
    # Add argument for evaluation class with specific choices
    args = parser.parse_args()
    adapter_path = args.adapter_path
    eval_path = args.eval_path

    evaluator = CombinedEvaluator()
    eval_logs = []
    eval_results = evaluator.empty_results()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        token=LLAMA_API_KEY
    )

    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path, token=LLAMA_API_KEY)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, token=LLAMA_API_KEY)

    # Set model to evaluation mode
    model.eval()
    input_data, labels, images = load_dataset(eval_path)
    total = len(input_data)
    for i in tqdm(range(total), desc="Running Eval"):
        prompt = input_data[i]
        image_fpath = images[i]
        image = Image.open(image_fpath).convert("RGB")
        label = labels[i]
        # Process input for model
        inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(output[0])
        response = response.strip().split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        response = response.split("<|eot_id|>")[0].strip()
        curr_eval_result = evaluator.evaluate(response, label)
        for key, value in curr_eval_result.items():
            eval_results[key].append(value)
        eval_logs.append(
            {
                "input": {
                    "image": image_fpath,
                    "prompt": prompt
                },
                "response": response,
                "label": label
            }
        )

    eval_keys = list(eval_results.keys())
    eval_counts = {key: len(eval_results[key]) for key in eval_keys}
    for key in eval_keys:
        if len(eval_results[key]) > 0:
            eval_results[key] = round(sum(eval_results[key]) / len(eval_results[key]), 3)
        else:
            eval_results[key] = 0.0
        print(f"{key}: {eval_results[key]}")
    # Report to 3 sf
    with open(f"./logs/eval_results.json", "w") as f:
        json_dump = {
            "results": eval_results,
            "eval_counts": eval_counts,
            "detailed_logs": eval_logs
        }
        json.dump(json_dump, f)
