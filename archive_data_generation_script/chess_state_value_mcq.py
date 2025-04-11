import os
import json
from utils.pgn_to_fen import pgn_to_fen
from utils.get_turn_from_fen import get_player_turn
import random
from concurrent.futures import ProcessPoolExecutor

SEED = 42
random.seed(SEED)

QUERY_TEMPLATES = [
    "In the following chess position, it is {turn}'s move. Which side has the advantage, or is the position equal?",
    "It is {turn} to play in this position. Assess the position and determine which side is in a better situation or if it's balanced.",
    "Now, {turn} is to move. Analyze the position and state whether {turn} has an advantage, or if the game is equal.",
    "{turn} is to play here. Who stands better in this position, or is it dynamically balanced?",
    "It's {turn}'s move at this stage. Evaluate the position and explain which side is ahead, or if it's equal.",
    "{turn} has the turn to play in this position. How would you assess the advantage—who is ahead, or is the position even?",
    "{turn} must make a move in this position. Determine if {turn} has an advantage, is at a disadvantage, or if the game is balanced.",
    "This position is {turn}'s to play. Which side appears to be in a stronger position, or is the evaluation equal?",
    "It's {turn}'s turn to play here. Who holds the upper hand in this position, or is it too close to call?",
    "At this moment, {turn} plays next. Assess the position—does one side have a clear advantage, or is the game equal?"
]


# TODO: Load file
def process_file(file_name: str):
    folder_dir = os.path.join(".", "archive", "chess_state_value_mcq")
    fpath = os.path.join(folder_dir, file_name)

    output = []

    with open(fpath, "r") as f:
        data = json.load(f)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_example, data["examples"]))

    output.extend(results)

    return output

def process_example(example):
    # TODO: Extract FEN from PGN
    board_fen = pgn_to_fen(example["input"])
    target = example["target"]
    turn = get_player_turn(board_fen[-1][0])
    query = random.choice(QUERY_TEMPLATES).format(turn = turn)
    return {
        "board_fen": board_fen[-1][0],
        "query": query,
        "target": target
    }


if __name__ == '__main__':
    output = process_file("chess_state_value_multi_choice_2.json")
    
    output_path = os.path.join(".", "archive", "chess_state_value_mcq", "chess_state_value_multi_choice_2_stage_1.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print("Data saved to", output_path)