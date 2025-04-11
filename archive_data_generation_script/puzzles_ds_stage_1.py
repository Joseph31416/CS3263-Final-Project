import os
import json
import multiprocessing
from utils import check_move_validity

TEXT = "text"
BOARD_PREFIX = "board's FEN is"
BOARD_SUFFIX = ","
SOL_PREFIX = "SAN format as"
SOL_SUFFIX = "and"

def get_all_themes() -> list[str]:
    fpath = os.path.join(".", "archive", "puzzles", "themes.json")
    with open(fpath, "r") as f:
        data = json.load(f)
    themes = data["themes"]
    return themes

def identify_themes(text: str, themes: list[str]):
    """
    Identify themes in the text.
    """
    text = text.lower()
    identified_themes = []
    for theme in themes:
        if theme.lower() in text:
            identified_themes.append(theme)
    return identified_themes

def process_file(idx):
    """
    Process a single file identified by its index.
    Returns a tuple: (list_of_valid_puzzles, failure_count, success_count)
    """
    folder_dir = os.path.join(".", "archive", "puzzles")
    fname_format = "chess_puzzle-data.jsonl-000{idx}-of-00016"
    fname = fname_format.format(idx=idx)
    fpath = os.path.join(folder_dir, fname)
    all_themes = get_all_themes()
    
    local_output = []
    local_failure = 0
    local_success = 0

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
                themes = identify_themes(text, all_themes)
                if len(themes) > 0:
                    local_output.append({"board_fen": board_fen, "solution": sol, "themes": themes})
                    local_success += 1
            else:
                local_failure += 1
        except Exception:
            local_failure += 1
    return local_output, local_failure, local_success

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
    output = []
    failure_count = 0
    success_count = 0
    
    for file_output, file_failure, file_success in results:
        output.extend(file_output)
        failure_count += file_failure
        success_count += file_success
    
    print(f"Number of failure cases: {failure_count}")
    print(f"Number of success cases: {success_count}")
    
    output_path = os.path.join(".", "archive", "puzzles", "puzzles_ds_stage_1_with_themes.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Data saved to {output_path}")
