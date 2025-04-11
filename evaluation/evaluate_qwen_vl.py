import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
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
    for row in data:
        content = []
        image = row["image"]
        content.append({"type": "image", "image": image})
        conversations = row["conversations"]
        human_conv = conversations[0]["value"]
        human_conv = human_conv.lstrip("<image>").strip()
        content.append({"type": "text", "text": human_conv})
        input_data.append(
            [
                {"role": "user", "content": content}
            ]
        )
        labels.append(conversations[1]["value"])
    return input_data, labels

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

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Load base model in 4-bit precision
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Set model to evaluation mode
    model.eval()
    input_data, labels = load_dataset(eval_path)
    total = len(input_data)
    for i in tqdm(range(total), desc="Running Eval"):
        messages = input_data[i]
        label = labels[i]
        # Process input for model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
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
    adapter_fname = adapter_path.strip("/").split("/")[-3]
    eval_output_fname = adapter_fname.split("_llava")[0].strip().split("train_")[-1].strip()
    with open(f"./logs/{eval_output_fname}.json", "w") as f:
        json_dump = {
            "results": eval_results,
            "eval_counts": eval_counts,
            "detailed_logs": eval_logs
        }
        json.dump(json_dump, f)
