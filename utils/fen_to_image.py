import chess
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def fen_to_matplotlib_chessboard(fen: str, output_file: str="chessboard_matplotlib.png"):
    """
    Draws a chessboard using matplotlib given a FEN string.
    
    The board uses the colors:
      - Dark squares: "#769656"
      - Light squares: "#EEEED2"
      
    Chess pieces are rendered using Unicode symbols.
    
    :param fen: FEN string representing the chess position.
    :param output_file: Filename for the saved PNG image.
    """
    board = chess.Board(fen=fen)
    
    # Define colors
    dark_color = "#769656"
    light_color = "#EEEED2"
    
    # Create a figure and axis
    _, ax = plt.subplots(figsize=(8, 8))
    
    # Draw board squares.
    # We'll use chess coordinates: file 0-7 and rank 0-7.
    # In chess, a1 is (file=0, rank=0) and a8 is (0,7). This means the
    # board will appear with white at the bottom (rank 1) and black at the top (rank 8).
    for file in range(8):
        for rank in range(8):
            # Choose square color: a8 (file 0, rank 7) should be dark.
            # With a1 at (0,0), the condition is: dark if (file + rank) is odd.
            square_color = light_color if (file + rank) % 2 == 1 else dark_color
            square = patches.Rectangle((file, rank), 1, 1, facecolor=square_color)
            ax.add_patch(square)
    
    # Mapping from piece symbols (as provided by python-chess) to Unicode chess symbols.
    piece_unicode = {
        "K": "\u2654", "Q": "\u2655", "R": "\u2656",
        "B": "\u2657", "N": "\u2658", "P": "\u2659",
        "k": "\u265A", "q": "\u265B", "r": "\u265C",
        "b": "\u265D", "n": "\u265E", "p": "\u265F"
    }
    
    # Place pieces on the board.
    # chess.square_file(sq) returns 0..7 for files a-h.
    # chess.square_rank(sq) returns 0..7 for ranks 1-8.
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            symbol = piece_unicode[piece.symbol()]
            # Place the text in the center of the square.
            ax.text(file + 0.5, rank + 0.5, symbol, fontsize=36,
                    ha='center', va='center')
    
    # Set the limits and aspect ratio so that squares are equal.
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Remove axis ticks and frame.
    ax.axis('off')
    
    # Optionally, add coordinate labels.
    # File labels (a-h) along the bottom.
    files_labels = list("abcdefgh")
    for file in range(8):
        ax.text(file + 0.5, -0.3, files_labels[file], ha='center', va='center', fontsize=12)
    # Rank labels (1-8) along the left side.
    for rank in range(8):
        # rank 0 corresponds to "1", rank 7 corresponds to "8".
        ax.text(-0.3, rank + 0.5, str(rank + 1), ha='center', va='center', fontsize=12)
    
    # Save the figure.
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
if __name__ == "__main__":
    # Example FEN. Adjust this string to generate a different position.
    example_fen = "rn1q1rk1/pp2bpp1/2p1b2p/8/3Pn2B/2N2P2/PPQ3PP/R3K1NR w KQ - 0 12"
    fen_to_matplotlib_chessboard(example_fen)
    print("Chessboard image saved as 'chessboard_matplotlib.png'")
