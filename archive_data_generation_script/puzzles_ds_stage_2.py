import os
import json
from utils import get_turn_from_fen
import random

SEED = 42
TEXT = "text"
BOARD_PREFIX = "board's FEN is"
BOARD_SUFFIX = ","
SOL_PREFIX = "SAN format as"
SOL_SUFFIX = "and"

random.seed(SEED)

QUERY_TEMPLATES = [
    "It is {turn} to play in this position. Give the winning sequence of moves for {turn} in UCI format.",
    "In this position, it is {turn}'s move. Provide the winning sequence of moves for {turn} in UCI format.",
    "It's {turn}'s turn to play here. Please list the winning moves for {turn} in UCI format.",
    "Now, {turn} is to move in this position. Show the sequence of winning moves for {turn} in UCI format.",
    "{turn} has the turn to play in this position. Give the correct winning sequence in UCI format.",
    "It is now {turn}'s move in this position. Provide the winning moves in UCI format.",
    "This position is {turn}'s to play. Specify the winning sequence in UCI format.",
    "{turn} must make a move in this position. Please outline the winning sequence in UCI format.",
    "It's {turn}'s move at this stage. Provide the correct winning move sequence in UCI format.",
    "{turn} is to play here. State the sequence of moves leading to victory in UCI format.",
    "At this moment, {turn} plays next. Write the winning move sequence in UCI format."
]

QUERY_TEMPLATES_WITH_THEME = [
    "It is {turn} to play in this position. First identify the theme(s) involved in the puzzle. Then, give the winning sequence of moves for {turn} in UCI format.",
    "In this position, it is {turn}'s move. What is the theme of the position? Provide the winning sequence of moves for {turn} in UCI format.",
    "It's {turn}'s turn to play here. What is the theme of the puzzle? Please list the winning moves for {turn} in UCI format.",
    "Now, {turn} is to move in this position. Give the theme(s) of the position and show the sequence of winning moves for {turn} in UCI format.",
    "{turn} has the turn to play in this position. List the theme(s) of the puzzle and give the correct winning sequence in UCI format.",
    "It is now {turn}'s move in this position. What theme(s) are involved in this position? Provide the winning moves in UCI format.",    
    "This position is {turn}'s to play. What is the theme of the position? Specify the winning sequence in UCI format.",
    "{turn} must make a move in this position. What is the theme of the puzzle? Please outline the winning sequence in UCI format.",
    "It's {turn}'s move at this stage. Consider the theme(s) of the position. Provide the correct winning move sequence in UCI format.",
    "{turn} is to play here. Identify the theme(s) of the position. State the sequence of moves leading to victory in UCI format.",
    "At this moment, {turn} plays next. What are the theme(s) of the position? Write the winning move sequence in UCI format."
]

def load_themes_mapping() -> dict[str, str]:
    fpath = os.path.join(".", "archive", "puzzles", "themes.json")
    with open(fpath, "r") as f:
        data = json.load(f)
    natural_form_mapping = data["natural_form_mapping"]
    return natural_form_mapping

def generate_train_test(num_train: int, num_test: int, include_themes: bool = False):
    folder_dir = os.path.join(".", "archive", "puzzles")
    fname = "puzzles_ds_stage_1_with_themes.json"
    output = []

    with open(os.path.join(folder_dir, fname), "r") as f:
        data = json.load(f)

    if num_train + num_test > len(data):
        raise ValueError(f"num_train + num_test exceeds the number of examples in the dataset. Only {len(data)} examples available.")
    # randomly sample num_train + num_test
    sampled_data = random.sample(data, num_train + num_test)
    natural_form_mapping = load_themes_mapping()
    for row in sampled_data:
        board_fen = row["board_fen"]
        sol = row["solution"]
        turn = get_turn_from_fen.get_player_turn(board_fen)
        if include_themes:
            query = random.choice(QUERY_TEMPLATES_WITH_THEME).format(turn=turn)
        else:
            query = random.choice(QUERY_TEMPLATES).format(turn=turn)
        target = sol.replace(",", ", ")
        if include_themes:
            themes = list(map(lambda x: natural_form_mapping[x], row["themes"]))
            if len(themes) == 1:
                target = f"The theme involved is {themes[0]}. The winning sequence is {target}."
            else:
                target = f"The themes involved are {', '.join(themes)}. The winning sequence is {target}."
        output.append({"board_fen": board_fen, "query": query, "target": target})

    random.shuffle(output)
    train_data = output[:num_train]
    test_data = output[num_train:]
    if include_themes:
        output_path_train = os.path.join(".", "data", "text", f"train_puzzles_with_themes_{num_train}.json")
        output_path_test = os.path.join(".", "data", "text", f"test_puzzles_with_themes_{num_test}_{num_train}.json")
    else:
        output_path_train = os.path.join(".", "data", "text", f"train_puzzles_{num_train}.json")
        output_path_test = os.path.join(".", "data", "text", f"test_puzzles_{num_test}.json")
    with open(output_path_train, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(output_path_test, "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"Training data saved to {output_path_train}")
    print(f"Number of training examples: {len(train_data)}")
    print(f"Test data saved to {output_path_test}")
    print(f"Number of test examples: {len(test_data)}")

if __name__ == "__main__":
    num_train = 100000
    num_test = 2000
    generate_train_test(num_train, num_test, include_themes=True)
