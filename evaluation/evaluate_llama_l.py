import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel
import argparse
import json
from tqdm import tqdm
from evaluator import CombinedEvaluator
from env.env import LLAMA_API_KEY

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_dataset(path: str) -> list[dict]:
    """
    Load the dataset from the given path.
    """
    with open(path, "r") as f:
        data = json.load(f)
    input_data = []
    labels = []
    for row in data:
        conversations = row["conversations"]
        human_conv = conversations[0]["value"]
        input_data.append(
            [
                {"role": "user", "content": human_conv}
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

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    token = LLAMA_API_KEY
    # Load base model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )

    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Load processor
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="<pad>"),
        tokenizer=tokenizer,
        model=model,
    )

    # Set model to evaluation mode
    model.eval()
    input_data, labels = load_dataset(eval_path)
    total = len(input_data)
    for i in tqdm(range(total), desc="Running Eval"):
        messages = input_data[i]
        label = labels[i]
        # Process input for model
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        # Generate response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    # sample adapter f_name: train_chess_annotation_fen_2500_llava_2025-03-28_01-30
    adapter_fname = adapter_path.split("/")[-3]
    eval_output_fname = adapter_fname.split("_llava")[0].strip().split("train_")[-1].strip()
    with open(f"./logs/{eval_output_fname}.json", "w") as f:
        json_dump = {
            "results": eval_results,
            "eval_counts": eval_counts,
            "detailed_logs": eval_logs
        }
        json.dump(json_dump, f)
