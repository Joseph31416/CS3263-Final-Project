import json
import os


def read_jsonl(file_path: str):
    """
    Reads a JSONL file and converts each entry into a dictionary.
    
    :param file_path: Path to the JSONL file.
    :return: List of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))  # Convert JSON string to dictionary
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    return data

if __name__ == "__main__":
    # Example usage:
    file_path = os.path.join(".", "archive", "annotated_pgn-data.jsonl-00000-of-00002")  
    json_data = read_jsonl(file_path)

    print(len(json_data))
    print(json_data[105]["text"])
