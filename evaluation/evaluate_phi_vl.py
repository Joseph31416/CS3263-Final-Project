import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import argparse
import json
from tqdm import tqdm
from evaluator import CombinedEvaluator

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
        content = []
        image_fpath = row["image"]
        images.append([image_fpath])
        conversations = row["conversations"]
        human_conv = conversations[0]["value"]
        human_conv = human_conv.strip().replace("<image>", "<|image_1|>")
        content.append({"type": "text", "text": human_conv})
        input_data.append(
            [
                {"role": "user", "content": human_conv}
            ]
        )
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

    model_id = "microsoft/Phi-3.5-vision-instruct"
    # Load base model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
         _attn_implementation='eager' 
    )

    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Set model to evaluation mode
    model.eval()
    input_data, labels, list_of_image_paths = load_dataset(eval_path)
    total = len(input_data)
    for i in tqdm(range(total), desc="Running Eval"):
        messages = input_data[i]
        label = labels[i]
        image_fpath = list_of_image_paths[i]
        images = [Image.open(image_fpath[0]).convert("RGB")]
        # Process input for model

        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = processor(prompt, images, return_tensors="pt").to("cuda") 

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0] 
        curr_eval_result = evaluator.evaluate(response, label)
        for key, value in curr_eval_result.items():
            eval_results[key].append(value)
        eval_logs.append(
            {
                "input": messages,
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
    adapter_fname = adapter_path.split("/")[-3]
    eval_output_fname = adapter_fname.split("_llava")[0].strip().split("train_")[-1].strip()
    with open(f"./logs/eval_{eval_output_fname}.json", "w") as f:
        json_dump = {
            "results": eval_results,
            "eval_counts": eval_counts,
            "detailed_logs": eval_logs
        }
        json.dump(json_dump, f)
