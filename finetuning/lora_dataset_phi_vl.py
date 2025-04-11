import copy
from typing import Dict
import torch
import transformers
import json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

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

def get_image_info(image_path, min_pixel, max_pixel):
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "image", 
                    "image": image_path,
                    "min_pixel": min_pixel,
                    "max_pixel": max_pixel
                }
            ]
        }
    ]
    image_input, _ = process_vision_info(messages)
    return image_input[0]

def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN, f"<|image_{count}|>", 1)
        count += 1

    return input_string, count

def llava_to_openai(conversations):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1  # Initialize image count here
    for conversation in conversations:
        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

class PhiVLSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        padding=True,
    ):
        super(PhiVLSupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        # rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.processor = processor
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        processor = self.processor
        if "image" in sources:
            image_files = sources["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            for image_file in image_files:
                images.append(Image.open(image_file).convert("RGB"))
        else:
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations']))

        all_input_ids = [torch.tensor([1])] # bos token id
        all_labels = [torch.tensor([-100])] # ignore bos token

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = processor.tokenizer.apply_chat_template([user_input], tokenize=False, add_generation_prompt=True)
            gpt_response = f"{gpt_response['content']}<|end|>\n"
            
            if idx == 0:
                inputs = processor(user_input, images, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                pixel_values = inputs.get('pixel_values')
                image_sizes = inputs.get('image_sizes')

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, return_tensors='pt')['input_ids']

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

        all_input_ids.append(torch.tensor([32000]))  # eos token id
        all_labels.append(torch.tensor([32000]))  # eos token id
        
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        
        if pixel_values is not None:
            data_dict.update(pixel_values=pixel_values, image_sizes=image_sizes)
        
        return data_dict

class DataCollatorForPhiVLSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_sizes = []
        
        for example in examples:
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_sizes.append(example["image_sizes"])
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_sizes = torch.cat(batch_image_sizes, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_sizes"] = image_sizes

        return data_dict
