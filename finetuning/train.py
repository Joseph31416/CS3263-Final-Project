from train_with_lora import train_with_lora
from config.config import Config
from utils.constants import QWEN_VL, LLAMA_VL, PHI_VL, QWEN_L, LLAMA_L, PHI_L
from env.env import LLAMA_API_KEY
from transformers.processing_utils import ProcessorMixin
from transformers.generation.utils import GenerationMixin
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, \
    BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import argparse
from typing import Tuple, Union

MODEL_IDS = [
    QWEN_VL, LLAMA_VL, PHI_VL,
    QWEN_L, LLAMA_L, PHI_L
]

def get_model_and_input_handler(model_id: str) -> Tuple[GenerationMixin, Union[ProcessorMixin, AutoTokenizer]]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    torch_dtype = torch.float16
    device_map = "auto"
    if model_id == QWEN_VL:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    QWEN_VL, torch_dtype=torch_dtype, device_map=device_map,
                    quantization_config=quantization_config
                )
        processor = AutoProcessor.from_pretrained(QWEN_VL)
        return model, processor
    elif model_id == LLAMA_VL:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            token=LLAMA_API_KEY
        )
        processor = AutoProcessor.from_pretrained(model_id, token=LLAMA_API_KEY)
        return model, processor
    elif model_id == PHI_VL:
        model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    device_map=device_map, 
                    trust_remote_code=True, 
                    torch_dtype=torch_dtype,
                    _attn_implementation='eager',
                    quantization_config=quantization_config
                )
        processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=4
        ) 
        return model, processor
    elif model_id in [QWEN_L, LLAMA_L, PHI_L]:
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    elif model_id == LLAMA_L:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            token=LLAMA_API_KEY
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=LLAMA_API_KEY)
        return model, tokenizer
    else:
        raise ValueError(f"Model ID {model_id} not recognized.")

def get_target_modules(model_id: str):
    TARGET_MODULES_MAPPING = {
        QWEN_VL: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        LLAMA_VL: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"],
        PHI_VL: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "out_proj"],
        QWEN_L: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        LLAMA_L: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        PHI_L: ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    }
    return TARGET_MODULES_MAPPING.get(model_id, None)

def get_lora_config(model_id: str):
    target_modules = get_target_modules(model_id)
    if target_modules is None:
        raise ValueError(f"Model ID {model_id} not recognized for LoRA configuration.")
    
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules
    )
    return lora_config

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_id", type=str, choices=MODEL_IDS, required=True, help="Model ID")
    parser.add_argument("--src_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--logging_steps", type=int, default=250, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    args = parser.parse_args()
    lora_config = get_lora_config(args.model_id)
    model, processor_or_tokenizer = get_model_and_input_handler(args.model_id)
    config = Config(
        model_id=args.model_id,
        model=model,
        processor=processor_or_tokenizer if args.model_id in [QWEN_VL, LLAMA_VL, PHI_VL] else None,
        tokenizer=processor_or_tokenizer if args.model_id in [QWEN_L, LLAMA_L, PHI_L] else None,
        src_path=args.src_path,
        eval_path=args.eval_path,
        num_epochs=args.num_epochs,
        checkpoint_path=args.checkpoint_path,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        lora_config=lora_config
    )
    train_with_lora(config)
