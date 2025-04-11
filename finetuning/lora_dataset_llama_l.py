import copy
import torch
import transformers
import json
from torch.utils.data import Dataset

IGNORE_INDEX = -100

START_TOKEN = "<|start_header_id|>"
END_TOKEN = "<|end_header_id|>"
BEGIN_OF_TEXT_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"
SYSTEM_TOKEN = "system"
USER_TOKEN = "user"
ASSISTANT_TOKEN = "assistant"

SYSTEM_MESSAGE = "Cutting Knowledge Date: December 2023.\nToday Date: 26 Jul 2024"

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conversation in conversations:
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": conversation["value"],
        }
        transformed_data.append(transformed_entry)
    return transformed_data

class LlamaLSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        tokenizer: transformers.PreTrainedTokenizer,
        padding=True,
    ):
        super(LlamaLSupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        tokenizer = self.tokenizer
        sources = copy.deepcopy(llava_to_openai(sources['conversations']))

        all_input_ids = [] 
        all_labels = []

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            gpt_prompt = f"{gpt_response['content']}{EOT_TOKEN}"
            if idx == 0:
                user_prompt = tokenizer.apply_chat_template([user_input], add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(user_prompt, add_special_tokens=False, return_tensors='pt')
                prompt_input_ids = inputs["input_ids"]
            response_input_ids = tokenizer(gpt_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict

class DataCollatorForLlamaLSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
                
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        
        labels = pad_sequence(
            batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX
        )
        
        attention_mask = input_ids != self.pad_token_id

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return batch_dict
