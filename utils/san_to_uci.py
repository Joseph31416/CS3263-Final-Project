import chess

def san_to_uci(fen: str, move_san: str) -> str:
    """Convert a SAN move to UCI format given a FEN representation."""
    board = chess.Board(fen)
    try:
        move = board.parse_san(move_san)
        return move.uci()
    except Exception:
        return None
