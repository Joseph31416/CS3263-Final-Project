import chess
import chess.pgn
import os
import io

def process_game(game: chess.pgn.Game, show_last_move: bool=False):
    board = game.board()
    output = []
    for move in game.mainline():
        move_number = board.fullmove_number
        comment = move.parent.comment if move.parent.comment else None
        board.push(move.move)
        if not show_last_move:
            output.append((board.fen(), comment))
        else:
            last_turn = "White" if move.turn() == chess.BLACK else "Black"
            output.append((board.fen(), comment, move.uci(), last_turn, move_number))
    return output

def pgn_to_fen(pgn_str: str, show_last_move: bool=False):
    pgn_io = io.StringIO(pgn_str)
    # Read the game from the string (instead of a file)
    game = chess.pgn.read_game(pgn_io)
    return process_game(game, show_last_move=show_last_move)

def file_pgn_to_fen(pgn_file: str, show_last_move: bool=False):
    with open(pgn_file, "r") as file:
        game = chess.pgn.read_game(file)  
    return process_game(game, show_last_move=show_last_move)

if __name__ == "__main__":
    "archive/scrap_data/15438/pgns"
    ann_id = "15438"
    gid = "1000846"
    pgn_file = os.path.join("archive", "scrap_data", ann_id, "pgns", f"{gid}.pgn")
    fens = file_pgn_to_fen(pgn_file, show_last_move=True)
    
    for i, el in enumerate(fens, 1):
        fen, comment, last_move, turn, move_number = el
        print(f"Move {i}: {fen} ({comment}) ({last_move}) ({turn}) ({move_number})")
