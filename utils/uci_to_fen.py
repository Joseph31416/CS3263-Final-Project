import chess
import chess.pgn

def uci_to_fen(moves, initial_fen=chess.STARTING_FEN):
    board = chess.Board(initial_fen)
    fen_list = []
    
    for move in moves.split():
        board.push_uci(move)
        fen_list.append({"Move": move, "FEN": board.fen()})
    
    return fen_list

if __name__ == '__main__':
    # Example usage
    moves = "e2e4 g7g6 d2d4 f8g7 c1e3 g8f6 f2f3 d7d6"
    fen = uci_to_fen(moves)
    for elem in fen:
        print(elem)
