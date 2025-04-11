import os
import json
from utils import pgn_to_fen
from utils import get_turn_from_fen
import random
from concurrent.futures import ProcessPoolExecutor

SEED = 42
random.seed(SEED)

QUERY_TEMPLATES = [
        "In the following chess position, it is {turn}'s move. Find a checkmate-in-one move in SAN format.",
        "It is {turn} to play in this position. Find a move in the chess position resulting in checkmate in SAN format.",
        "Now, {turn} is to move in this position. Can you find the checkmate-in-one move in the following chess position in SAN format?",
        "{turn} is to play here. What is the move that results in checkmate in one move in this position in SAN format?",
        "It's {turn}'s move at this stage. Identify the checkmate-in-one move in this chess position in SAN format.",
        "{turn} has the turn to play in this position. What is the move in SAN format that delivers checkmate in one in this position?",
        "{turn} must make a move in this position. Find the move that results in a checkmate-in-one in the following chess position in SAN format.",
        "This position is {turn}'s to play. Which move leads to checkmate in one move in this position in SAN format?",
        "It's {turn}'s turn to play here. Can you spot the checkmate-in-one move in the current chess position in SAN format?",
        "At this moment, {turn} plays next. What is the checkmate-in-one move in this particular chess arrangement in SAN format?"
    ]

def process_example(example):
    board_fen = pgn_to_fen.pgn_to_fen(example["input"])
    target = example["target"]

    turn = get_turn_from_fen.get_player_turn(board_fen[-1][0])
    query = random.choice(QUERY_TEMPLATES).format(turn=turn)
    return {
        "board_fen": board_fen[-1][0],
        "query": query,
        "target": target
    }


def process_file(file_name: str) -> tuple:
    """
    Process a single file identified by its index.
    Returns a tuple: (list_of_valid_puzzles, failure_count, success_count)
    """
    folder_dir = os.path.join(".", "archive", "checkmate_in_one")
    fpath = os.path.join(folder_dir, file_name)

    output = []

    with open(fpath, "r") as f:
        data = json.load(f)

    # Use ProcessPoolExecutor to parallelize the loop over data["examples"]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_example, data["examples"]))

    # Append the results to output
    output.extend(results)

    return output


if __name__ == '__main__':
    output = process_file("checkmate_in_one_ori.json")
    
    output_path = os.path.join(".", "archive", "checkmate_in_one", "checkmate_in_one_stage_1.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print("Data saved to", output_path)
