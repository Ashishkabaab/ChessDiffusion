import chess
import numpy as np

# -----------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------
# Squares: 0=empty, 1=wK, 2=wQ, 3=wR, 4=wB, 5=wN, 6=wP
#          7=bK, 8=bQ, 9=bR, 10=bB, 11=bN, 12=bP
# Token 64: side to move (0=white, 1=black)
# Token 65: castling K  (0/1)
# Token 66: castling Q  (0/1)
# Token 67: castling k  (0/1)
# Token 68: castling q  (0/1)
# Token 69: en passant  (0=none, 1-8 = file a-h)
# Token 70: halfmove clock (capped at 50)
# Token 71: fullmove counter (capped at 150)
# -----------------------------------------------------------

SEQ_LEN = 72
VOCAB_SIZE = 13  # for square tokens (0-12)

PIECE_TO_IDX = {
    (chess.KING,   chess.WHITE): 1,
    (chess.QUEEN,  chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.BISHOP, chess.WHITE): 4,
    (chess.KNIGHT, chess.WHITE): 5,
    (chess.PAWN,   chess.WHITE): 6,
    (chess.KING,   chess.BLACK): 7,
    (chess.QUEEN,  chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.BISHOP, chess.BLACK): 10,
    (chess.KNIGHT, chess.BLACK): 11,
    (chess.PAWN,   chess.BLACK): 12,
}

IDX_TO_PIECE = {v: k for k, v in PIECE_TO_IDX.items()}


def fen_to_tokens(fen: str) -> np.ndarray:
    """Convert a FEN string to a 72-token integer array."""
    try:
        board = chess.Board(fen)
    except Exception:
        return None

    tokens = np.zeros(SEQ_LEN, dtype=np.int64)

    # --- 64 square tokens ---
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            tokens[sq] = PIECE_TO_IDX[(piece.piece_type, piece.color)]
        else:
            tokens[sq] = 0

    # --- token 64: side to move ---
    tokens[64] = 0 if board.turn == chess.WHITE else 1

    # --- tokens 65-68: castling rights ---
    tokens[65] = int(board.has_kingside_castling_rights(chess.WHITE))
    tokens[66] = int(board.has_queenside_castling_rights(chess.WHITE))
    tokens[67] = int(board.has_kingside_castling_rights(chess.BLACK))
    tokens[68] = int(board.has_queenside_castling_rights(chess.BLACK))

    # --- token 69: en passant ---
    if board.ep_square is not None:
        tokens[69] = chess.square_file(board.ep_square) + 1  # 1-8
    else:
        tokens[69] = 0

    # --- token 70: halfmove clock (capped at 50) ---
    tokens[70] = min(board.halfmove_clock, 50)

    # --- token 71: fullmove counter (capped at 150) ---
    tokens[71] = min(board.fullmove_number, 150)

    return tokens


def tokens_to_fen(tokens: np.ndarray) -> str:
    """Convert a 72-token array back to a FEN string."""
    board = chess.Board(fen=None)
    board.clear()

    # --- place pieces on squares ---
    for sq in range(64):
        idx = int(tokens[sq])
        if idx != 0 and idx in IDX_TO_PIECE:
            piece_type, color = IDX_TO_PIECE[idx]
            board.set_piece_at(sq, chess.Piece(piece_type, color))

    # --- side to move ---
    board.turn = chess.WHITE if int(tokens[64]) == 0 else chess.BLACK

    # --- castling rights ---
    castling = ""
    if tokens[65]: castling += "K"
    if tokens[66]: castling += "Q"
    if tokens[67]: castling += "k"
    if tokens[68]: castling += "q"
    board.set_castling_fen(castling if castling else "-")

    # --- en passant ---
    ep = int(tokens[69])
    if ep != 0:
        rank = 5 if board.turn == chess.BLACK else 2
        board.ep_square = chess.square(ep - 1, rank)
    else:
        board.ep_square = None

    # --- halfmove clock and fullmove counter ---
    board.halfmove_clock = int(tokens[70])
    board.fullmove_number = max(1, int(tokens[71]))

    return board.fen()


if __name__ == "__main__":
    # quick sanity check
    test_fens = [
        "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27",
        "8/4R3/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 w - - 1 64",
    ]

    print("Testing tokenizer round-trip...\n")
    for fen in test_fens:
        tokens = fen_to_tokens(fen)
        recovered = tokens_to_fen(tokens)
        board_orig = chess.Board(fen)
        board_recv = chess.Board(recovered)
        match = board_orig.board_fen() == board_recv.board_fen()
        print(f"Original:  {fen}")
        print(f"Recovered: {recovered}")
        print(f"Board match: {match}")
        print(f"Tokens: {tokens}")
        print()
