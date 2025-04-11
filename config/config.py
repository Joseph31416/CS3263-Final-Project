from peft import LoraConfig

class Config:

    def __init__(self, **kwargs):
        self.model_id = kwargs.get("model_id", None)
        self.model = kwargs.get("model", None)
        self.processor = kwargs.get("processor", None)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.src_path = kwargs.get("src_path", None)
        self.eval_path = kwargs.get("eval_path", None)
        self.num_epochs = kwargs.get("num_epochs", 3)
        self.checkpoint_path = kwargs.get("checkpoint_path", None)
        self.lora_config: LoraConfig = kwargs.get("lora_config", None)
        self.logging_strategy = kwargs.get("logging_strategy", "steps")
        self.logging_steps = kwargs.get("logging_steps", 250)
        self.eval_steps = kwargs.get("eval_steps", 500)
