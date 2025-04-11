import os
import json
from utils import uci_to_fen
import random

SEED = 42
random.seed(SEED)

QUERY_TEMPLATES = [
    "In this chess game, please give all the legal moves for the chess piece in the position {next_start_position} in UCI format, with each move separated by \\",
    "List all legal moves for the chess piece at {next_start_position} in UCI format, separating each move with \\",
    "Can you provide all valid moves in UCI format for the piece at {next_start_position}? Separate each move with \\",
    "Show me every possible legal move in UCI notation for the chess piece at {next_start_position}, using \\ as a separator.",
    "What are all the allowed moves for the piece at {next_start_position} in UCI format? Separate each move with \\",
    "Give me all possible moves in UCI notation for the piece at {next_start_position}, with moves delimited by \\",
    "Generate a list of all legal moves for the chess piece at {next_start_position} in UCI format, separating moves with \\",
    "Which UCI moves can the piece at {next_start_position} make? Format each move separated by \\",
    "Provide all valid UCI-formatted moves for the piece at {next_start_position}, ensuring each move is separated by \\",
    "Tell me every possible move in UCI format for the piece at {next_start_position}, using \\ as a separator.",
    "What are the available UCI moves for the chess piece at {next_start_position}? Split each move with \\"
]

def process_example(example):
    """
    Process a single example.
    Returns a tuple: (board_fen, next_start_position)
    """
    board_uci = example["input"][0:-3].strip()
    next_start_position = example["input"][-3:].strip()
    board_fen = uci_to_fen.uci_to_fen(board_uci)[-1]["FEN"]
    # print("board_uci: ", board_uci)
    # print("next_start_position: ", next_start_position)
    # print("board_fen: ", board_fen)

    return board_fen, next_start_position

def process_target(target, next_start_position):
    """
    Process a list of targets.
    Returns a list of legal moves in UCI format.
    """
    target.sort()
    # print("target: ", target)
    result = "\\".join([next_start_position + move for move in target])
    # print("result: ", result)
    return result

def process_file(file_name: str):

    folder_dir = os.path.join(".", "archive", "state_tracking")
    fpath = os.path.join(folder_dir, file_name)

    output = []

    with open(fpath, "r") as f:
        data = json.load(f)

    # print("len_data_example: ", len(data["examples"]))

    for example in data["examples"]:
        board_fen, next_start_position = process_example(example)
        target = process_target(example["target"], next_start_position)
        query = random.choice(QUERY_TEMPLATES).format(next_start_position=next_start_position)

        output.append({
            "board_fen": board_fen,
            "query": query,
            "target": target
        })

    return output



if __name__ == '__main__':
    fnames = [
        "real_short_ori.json", "real_medium_ori.json", "real_long_ori.json",
        "syn_short_ori.json", "syn_medium_ori.json", "syn_long_ori.json"
    ]
    output = []
    for fname in fnames:
        output.extend(process_file(fname))

    output_path = os.path.join(".", "archive", "state_tracking", "state_tracking_stage_1.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print("Data saved to", output_path)
    print(f"Total examples: {len(output)}")