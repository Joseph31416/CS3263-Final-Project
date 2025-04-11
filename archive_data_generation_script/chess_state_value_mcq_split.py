import os
import json
import random

SEED = 42
random.seed(SEED)

def generate_train_test(num_train: int, num_test: int):
    folder_dir = os.path.join(".", "archive", "chess_state_value_mcq")
    fname = "chess_state_value_multi_choice_2_stage_1.json"

    with open(os.path.join(folder_dir, fname), "r") as f:
        data = json.load(f)

    random.shuffle(data)
    if num_train + num_test > len(data):
        raise ValueError(f"num_train + num_test exceeds the number of examples in the dataset. Only {len(data)} examples available.")
    train_data = data[:num_train]
    test_data = data[num_train:num_train + num_test]
    output_path_train = os.path.join(".", "data", "text", f"train_chess_state_value_mcq_{num_train}.json")
    output_path_test = os.path.join(".", "data", "text", f"test_chess_state_value_mcq_{num_test}.json")
    with open(output_path_train, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(output_path_test, "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"Training data saved to {output_path_train}")
    print(f"Number of training examples: {len(train_data)}")
    print(f"Test data saved to {output_path_test}")
    print(f"Number of test examples: {len(test_data)}")

if __name__ == "__main__":
    num_train = 1000
    num_test = 200
    generate_train_test(num_train, num_test)