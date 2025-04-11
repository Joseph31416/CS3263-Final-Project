import os
from transformers import TrainingArguments, Trainer
from utils.constants import QWEN_VL, LLAMA_VL, PHI_VL, QWEN_L, LLAMA_L, PHI_L, GMT_8, TIME_FORMAT, LORA_OUTPUT_DIR
from utils.make_dir_safe import make_directory_safe
from peft import get_peft_model, prepare_model_for_kbit_training
from lora_dataset import make_supervised_data_module
import os
from datetime import datetime
from config.config import Config

os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

def train_with_lora(config: Config):
    src_path = config.src_path
    task_name = src_path.strip("/").split("/")[-2]
    eval_path = config.eval_path
    num_epochs = config.num_epochs
    checkpoint_path = config.checkpoint_path
    model_id = config.model_id
    logging_strategy = config.logging_strategy
    logging_steps = config.logging_steps
    eval_steps = config.eval_steps
    model = config.model
    processor = config.processor
    tokenizer = config.tokenizer

    lora_config = config.lora_config
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if model_id in [QWEN_VL, LLAMA_VL, PHI_VL]:
        data_module = make_supervised_data_module(model_id, processor, src_path, eval_data_path=eval_path)
    elif model_id in [QWEN_L, LLAMA_L, PHI_L]:
        data_module = make_supervised_data_module(model_id, tokenizer, src_path, eval_data_path=eval_path)

    # Training arguments
    current_time = datetime.now(GMT_8).strftime(TIME_FORMAT)
    current_time = current_time.replace(" ", "_").replace(":", "-")
    if checkpoint_path is None:
        output_dir = os.path.join(LORA_OUTPUT_DIR, make_directory_safe(model_id), f"{task_name}_{current_time}")
    else:
        output_dir = "/".join(checkpoint_path.strip().split("/")[:-2])
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_dir_training_args = os.path.join(output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir_training_args,
        per_device_train_batch_size=4, # 4 per device is the max on H100 47GB
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=num_epochs,
        logging_dir="./logs",
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        save_strategy=logging_strategy,
        save_steps=logging_steps,
        eval_strategy=logging_strategy,
        eval_steps=eval_steps,
        report_to=None,
        fp16=True,
        optim="paged_adamw_8bit"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],  # Ensure evaluation dataset is provided
        data_collator=data_module["data_collator"],
    )
    trainer.can_return_loss = True

    # Train model
    if type(checkpoint_path) == str and len(checkpoint_path) > 0:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # Save LoRA adapter
    lora_model_output_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(lora_model_output_dir)
