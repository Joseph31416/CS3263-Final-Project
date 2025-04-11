import chess

def get_player_turn(fen: str) -> str:
    """Determine the player's turn from a FEN representation."""
    board = chess.Board(fen)
    return "White" if board.turn == chess.WHITE else "Black"

if __name__ == "__main__":
    # Example usage
    fen_string = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq"
    print(f"It is {get_player_turn(fen_string)}'s turn.")
