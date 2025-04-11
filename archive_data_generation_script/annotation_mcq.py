import os
import json
from utils.pgn_to_fen import pgn_to_fen
import random
from concurrent.futures import ProcessPoolExecutor

SEED = 42
random.seed(SEED)

QUERY_TEMPLATES = [
    "From the options given, choose the annotation that matches the given position best. It is move {move_number} and {last_turn} made the move {last_move} on their last move.",
    "Select the annotation that best describes the current position. It's move {move_number}, and {last_turn} just played {last_move}.",
    "Given the position, pick the annotation that fits best. {last_turn} made the last move {last_move} on move {move_number}.",
    "It's move {move_number}, and {last_turn} has just completed their move {last_move}. Choose the annotation from the given options that matches the position most accurately.",
    "At move {move_number}, {last_turn} just played the move {last_move}. Which annotation best fits this position?",
    "This is move {move_number}, with {last_turn} having just played {last_move}. Select the most appropriate annotation for the position.",
    "It is move {move_number} and {last_turn} made the latest move {last_move}, which annotation aligns best with the position?",
    "Choose the annotation that best corresponds to the current board. The last move {last_move} was by {last_turn} on move {move_number}.",
    "It's now move {move_number}, and {last_turn} played {last_move}. Pick the annotation that reflects the position best.",
    "It is move {move_number} as of now, {last_turn} has just made the move {last_move}. Identify the annotation that most accurately represents this position."
]

options_dist = {}

def process_file(file_name: str):
    folder_dir = os.path.join(os.getcwd(), "archive", "chess_annotation")
    fpath = os.path.join(folder_dir, file_name)

    output = []

    with open(fpath, "r") as f:
        data = json.load(f)

    # data["examples"] is a list of length 3000, each of which is a dictionary
    # structure of said dict: {"input": pgn - str, "target_scores": {option - str: 0/1 - score}}

    # count_options(data["examples"])

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_example, data["examples"]))

    output.extend(results)

    return output


def count_options(data):
    dist = {}
    for example in data:
        target = example["target_scores"]
        count = 1
        for item in target:
            if target[item] == 1:
                if count not in dist:
                    dist[count] = 0
                dist[count] += 1
                break
            count += 1

    print(dist)


def process_example(example):
    # TODO: Extract FEN from PGN
    board_data = pgn_to_fen(example["input"], show_last_move=True)
    # only need the latest board position is of interest
    board_fen, _, last_move, last_turn, move_number = board_data[-1]
    target = example["target_scores"]
        
    keys = list(target.keys())
    random.shuffle(keys)
    query = random.choice(QUERY_TEMPLATES).format(move_number=move_number, last_turn=last_turn, last_move=last_move)
    count = 1
    correct_option = ""
    options_str = ""
    for item in keys:
        options_str += f"\n{count}. {item}"
        if target[item] == 1:
            correct_option = f"{count}. {item}"
        count += 1
    query = options_str + "\n\n" + query
    query = query.strip()
    return {
        "board_fen": board_fen,
        "query": query,
        "target": correct_option
    }


if __name__ == '__main__':
    output = process_file("chess_annotation.json")
    
    output_path = os.path.join(os.getcwd(), "archive", "chess_annotation", "chess_annotation_stage_1.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print("Data saved to", output_path)
    print(f"Number of examples: {len(output)}")