import argparse
import json
import os
import random
from multiprocessing import Pool
from utils import fen_to_image

HUMAN = "human"
GPT = "gpt"
SEED = 42
random.seed(SEED)

def process_row(args):
    input_fname, idx, row, output_folder_path = args
    id_ = f"{input_fname}_{idx}"
    board_fen = row["board_fen"]
    image_path = os.path.join(output_folder_path, f"{id_}.png")
    fen_to_image.fen_to_matplotlib_chessboard(board_fen, image_path)
    conversations = [
        {
            "from": HUMAN,
            "value": f"<image>\n{row['query']}"
        },
        {
            "from": GPT,
            "value": row["target"]
        }
    ]
    return {"id": id_, "image": image_path, "conversations": conversations}

def convert_entries_to_llava(src_path: str, num_cores: int = 4):
    # Use the folder name of src_path as folder name.
    folder_name = src_path.strip("/").split("/")[-2]
    fname = src_path.strip("/").split("/")[-1]
    output_image_folder_path = os.path.join("data", "images", folder_name)
    output_text_folder_path = os.path.join("data", "text", folder_name)
    os.makedirs(output_image_folder_path, exist_ok=True)
    os.makedirs(output_text_folder_path, exist_ok=True)
    
    with open(src_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    
    # Prepare tasks for each row.
    tasks = [(folder_name, idx, row, output_image_folder_path) for idx, row in enumerate(data)]
    
    # Use multiprocessing to process rows in parallel.
    with Pool(processes=num_cores) as pool:
        output = pool.map(process_row, tasks)
    
    # sort output by id
    # print(output[0])
    output = sorted(output, key=lambda x: int(x["id"].split("_")[-1]))
    
    output_path = os.path.join(output_text_folder_path, fname)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Image data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text data to image data.")
    parser.add_argument("--src_path", type=str, help="Path to the source JSON file, relative to the project root.")
    parser.add_argument("--num_cores", type=int, default=1, help="Number of cores to use for parallel processing.")
    args = parser.parse_args()
    src_path = args.src_path
    num_cores = args.num_cores
    convert_entries_to_llava(src_path=src_path, num_cores=num_cores)
