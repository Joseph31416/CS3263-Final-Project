import chess

def is_valid_moves(fen: str, list_move_san: list[str]):
    """Check if a move in SAN format is valid and update the board if it is."""
    try:
        board = chess.Board(fen)
        for move_san in list_move_san:
            move = board.parse_san(move_san)  # Parse move in SAN format
            if move in board.legal_moves:
                board.push(move)  # Make the move
            else:
                return False
        return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    fen = "r2qk2r/1pp3pp/p1nb1p2/4p3/3P2b1/1BP2N1n/PP3PK1/RNBQR3 w kq - 1 14"
    moves = "f3e5, g4d1, e5c6, e8d7, c6d8"
    list_moves = [el.strip() for el in moves.split(", ")]
    print(is_valid_moves(fen, list_moves))
