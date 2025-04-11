import os
import json
import multiprocessing
from utils import check_move_validity

TEXT = "text"
BOARD_PREFIX = "board's FEN is"
BOARD_SUFFIX = ","
SOL_PREFIX = "SAN format as"
SOL_SUFFIX = "and"
prefixes = ["theme of", "focuses on"]

def process_file(idx):
    """
    Process a single file identified by its index.
    Returns a tuple: (list_of_valid_puzzles, failure_count, success_count)
    """
    folder_dir = os.path.join(".", "archive", "puzzles")
    fname_format = "chess_puzzle-data.jsonl-000{idx}-of-00016"
    fname = fname_format.format(idx=idx)
    fpath = os.path.join(folder_dir, fname)

    themes_found = set()

    with open(fpath, "r") as f:
        data = f.readlines()
    
    for row in data:
        row = json.loads(row)
        text = row[TEXT].strip()
        try:
            # Extract board FEN and solution text
            board_fen = text.split(BOARD_PREFIX)[1].split(BOARD_SUFFIX)[0].strip()
            sol = text.split(SOL_PREFIX)[1].split(SOL_SUFFIX)[0].strip()
            
            # Get list of moves in SAN format
            list_san_moves = [el.strip() for el in sol.split(",")]
            if check_move_validity.is_valid_moves(board_fen, list_san_moves):
                for theme_prefix in prefixes:
                    if theme_prefix in text:
                        themes = text.split(theme_prefix)[1].strip().split()[0].strip(",").split(",")
                        for theme in themes:
                            themes_found.add(theme.strip("."))
                        break
        except Exception:
            continue
    return sorted(list(themes_found))

if __name__ == '__main__':
    indexes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    
    # Create a pool with as many processes as there are CPU cores
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Map the indexes to the process_file function in parallel
    results = pool.map(process_file, indexes)
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    # Aggregate results from all processes
    themes_found = set()
    failure_count = 0
    success_count = 0
    
    for local_themes_found in results:
        for theme in local_themes_found:
            themes_found.add(theme)
    
    # print(f"Number of failure cases: {failure_count}")
    # print(f"Number of success cases: {success_count}")
    
    output_path = os.path.join(".", "archive", "puzzles", "themes.json")
    output = {"themes": sorted(list(themes_found))}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Data saved to {output_path}")
