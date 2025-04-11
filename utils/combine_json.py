import json
import argparse
from typing import List

def combine(list_of_paths: List[str], output_path: str):
    """
    Combines multiple JSOL files into a single JSON file.
    
    :param list_of_paths: List of paths to JSON files.
    :param output_dir: Path to the output directory.
    """
    combined_data = []
    for file_path in list_of_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
        combined_data.extend(data)
    
    with open(output_path, 'w') as file:
        json.dump(combined_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple JSON files into a single JSON file.")
    parser.add_argument("--list_of_paths", type=str, nargs="+", help="List of paths to JSON files.")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()
    
    combine(args.list_of_paths, args.output_path)
