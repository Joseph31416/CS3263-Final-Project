import transformers
from typing import Union, Dict
from lora_dataset_qwen_vl import QwenVLSupervisedDataset, DataCollatorForQwenVLSupervisedDataset
from lora_dataset_llama_vl import LlamaVLSupervisedDataset, DataCollatorForLlamaVLSupervisedDataset
from lora_dataset_phi_vl import PhiVLSupervisedDataset, DataCollatorForPhiVLSupervisedDataset
from lora_dataset_qwen_l import QwenLSupervisedDataset, DataCollatorForQwenLSupervisedDataset
from lora_dataset_llama_l import LlamaLSupervisedDataset, DataCollatorForLlamaLSupervisedDataset
from lora_dataset_phi_l import PhiLSupervisedDataset, DataCollatorForPhiLSupervisedDataset
from utils.constants import QWEN_VL, LLAMA_VL, PHI_VL, QWEN_L, LLAMA_L, PHI_L

DATASET_COLLATER_MAPPING = {
    QWEN_VL: (QwenVLSupervisedDataset, DataCollatorForQwenVLSupervisedDataset),
    LLAMA_VL: (LlamaVLSupervisedDataset, DataCollatorForLlamaVLSupervisedDataset),
    PHI_VL: (PhiVLSupervisedDataset, DataCollatorForPhiVLSupervisedDataset),
    QWEN_L: (QwenLSupervisedDataset, DataCollatorForQwenLSupervisedDataset),
    LLAMA_L: (LlamaLSupervisedDataset, DataCollatorForLlamaLSupervisedDataset),
    PHI_L: (PhiLSupervisedDataset, DataCollatorForPhiLSupervisedDataset)
}

def make_supervised_data_module(
        model_id: str, processor_or_tokenizer: Union[transformers.ProcessorMixin, transformers.PreTrainedTokenizer],
        train_data_path: str, eval_data_path:str=None
    ) -> Dict:
    """Make datasets and collator for supervised fine-tuning.
    
    Args:
        processor: The processor used for tokenization and image processing.
        train_data_path: Path or list for training data.
        eval_data_path: Optional; path or list for evaluation data.
        
    Returns:
        A dictionary with keys 'train_dataset', 'eval_dataset', and 'data_collator'.
    """
    DatasetCls, DataCollatorCls = DATASET_COLLATER_MAPPING.get(model_id, (None, None))
    if DatasetCls is None or DataCollatorCls is None:
        raise ValueError(f"Unsupported model_id: {model_id}. Supported ids are: {list(DATASET_COLLATER_MAPPING.keys())}")

    if model_id in [QWEN_VL, LLAMA_VL, PHI_VL]:
        train_dataset = DatasetCls(
            data_path=train_data_path, processor=processor_or_tokenizer
        )
        
        eval_dataset = None
        if eval_data_path is not None:
            eval_dataset = DatasetCls(
                data_path=eval_data_path, processor=processor_or_tokenizer
            )
        
        data_collator = DataCollatorCls(pad_token_id=processor_or_tokenizer.tokenizer.pad_token_id)
    elif model_id in [QWEN_L, LLAMA_L, PHI_L]:
        train_dataset = DatasetCls(
            data_path=train_data_path, tokenizer=processor_or_tokenizer
        )
        
        eval_dataset = None
        if eval_data_path is not None:
            eval_dataset = DatasetCls(
                data_path=eval_data_path, tokenizer=processor_or_tokenizer
            )
        
        data_collator = DataCollatorCls(pad_token_id=processor_or_tokenizer.pad_token_id)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
