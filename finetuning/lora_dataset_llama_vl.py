import copy
import torch
import transformers
import json
from PIL import Image
from torch.utils.data import Dataset
import re

IGNORE_INDEX = -100

START_HEADER_TOKEN = "<|start_header_id|>"
END_HEADER_TOKEN = "<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"


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

def pad_cross_attention_mask(cross_attention_masks):
    shapes = [cam.shape for cam in cross_attention_masks]
    max_len = max(s[1] for s in shapes)     
    max_num_images = max(s[2] for s in shapes)  
    max_num_tiles = max(s[3] for s in shapes)

    batch_cam = torch.zeros(
        (len(cross_attention_masks), max_len, max_num_images, max_num_tiles),
        dtype=cross_attention_masks[0].dtype,
        device=cross_attention_masks[0].device
    )

    for i, cam in enumerate(cross_attention_masks):
        _, L, N, T = cam.shape
        batch_cam[i, :L, :N, :T] = cam

    return batch_cam

class LlamaVLSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        padding=True,
    ):
        super(LlamaVLSupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.padding = padding

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None

        processor = self.processor
        if "image" in sources:
            is_dummy = False
            image_files = sources["image"]

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
        
            for image_file in image_files:
                images.append(Image.open(image_file).convert("RGB"))

        else:
            is_dummy = True
            images = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

        all_input_ids = [] 
        all_labels = []

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            gpt_prompt = f"{gpt_response['content'][0]['text']}{EOT_TOKEN}"
            if idx == 0:
                # print(f"===User input===")
                # print(user_input)
                user_prompt = processor.apply_chat_template([user_input], add_generation_prompt=True)
                # print(f"===User prompt===")
                # print(user_prompt)
                if images is not None:
                    inputs = processor(images, user_prompt, add_special_tokens=False, return_tensors='pt')
                    pixel_values = inputs['pixel_values']
                    aspect_ratio_mask = inputs['aspect_ratio_mask']
                    aspect_ratio_ids = inputs['aspect_ratio_ids']
                    cross_attention_mask = inputs['cross_attention_mask']

                prompt_input_ids = inputs["input_ids"]

            response_input_ids = processor.tokenizer(gpt_prompt, add_special_tokens=False, return_tensors='pt')['input_ids']

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

        B, old_len, N, T = cross_attention_mask.shape
        if is_dummy:
            new_cross_attention_mask = torch.zeros((B, len(input_ids), N, T), dtype=cross_attention_mask.dtype)
            new_cross_attention_mask[:, :old_len, :, :] = cross_attention_mask
        else:
            new_cross_attention_mask = torch.ones((B, len(input_ids), N, T), dtype=cross_attention_mask.dtype)
            new_cross_attention_mask[:, :old_len, :, :] = cross_attention_mask

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            cross_attention_mask=new_cross_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return data_dict

class DataCollatorForLlamaVLSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_aspect_ratio_ids = []
        batch_aspect_ratio_mask = []
        batch_cross_attention_mask = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example.get("pixel_values"))
            batch_aspect_ratio_ids.append(example.get("aspect_ratio_ids"))
            batch_aspect_ratio_mask.append(example.get("aspect_ratio_mask"))
            batch_cross_attention_mask.append(example.get("cross_attention_mask"))
                
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        
        labels = pad_sequence(
            batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX
        )

        cross_attention_mask = pad_cross_attention_mask(batch_cross_attention_mask)
        
        attention_mask = input_ids != self.pad_token_id
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        aspect_ratio_ids = torch.cat(batch_aspect_ratio_ids, dim=0)
        aspect_ratio_mask = torch.cat(batch_aspect_ratio_mask, dim=0)


        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        batch_dict['pixel_values'] = pixel_values
        batch_dict['aspect_ratio_ids'] = aspect_ratio_ids
        batch_dict['aspect_ratio_mask'] = aspect_ratio_mask
        batch_dict['cross_attention_mask'] = cross_attention_mask

        return batch_dict

def replace_image_tokens(input_string, start_count=0):

    pattern = re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
    
    matches = re.findall(pattern, input_string)
    has_image = bool(matches)
    
    output_string = re.sub(pattern, '', input_string)
    
    new_count = start_count + len(matches)
    
    return output_string, new_count, has_image

def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)
    return input_string

def llava_to_openai(conversations, is_video=False, num_frames=None):

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 0
    for conversation in conversations:
        # if is_video:
        #     conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)
        
        transformed_content, image_count, has_image = replace_image_tokens(conversation["value"], image_count)
        content = []
        if has_image:
            for _ in range(image_count):
                content.append({"type":"image"})
        content.append({"type":"text", "text":transformed_content})
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data
