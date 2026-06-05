import sys
sys.path.insert(0, '/project/d3pm')

import torch
import numpy as np
import chess
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from d3pm_runner import D3PM
from dit import DDiT_Llama
from tokenizer import tokens_to_fen, fen_to_tokens

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CHECKPOINT_PATH = '/project/checkpoints/d3pm_fen_final.pth'
TOKENS_PATH     = '/project/tokens.npy'
OUTPUT_DIR      = '/project/evaluation'
N_GENERATE      = 1000
BATCH_SIZE      = 64
SEQ_LEN         = 72
VOCAB_SIZE      = 151
N_T             = 1000
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
def load_model():
    print("Loading model...")
    model = DDiT_Llama(VOCAB_SIZE, dim=512, n_layers=6)
    d3pm = D3PM(model, n_T=N_T, num_classes=VOCAB_SIZE, hybrid_loss_coeff=0.01).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    d3pm.load_state_dict(ckpt['model_state_dict'])
    d3pm.eval()
    print(f"Model loaded from step {ckpt['step']}")
    return d3pm

# -----------------------------------------------------------
# Generate FENs
# -----------------------------------------------------------
def generate_fens(d3pm, n=N_GENERATE):
    print(f"Generating {n} FENs...")
    all_tokens = []
    with torch.no_grad():
        for i in tqdm(range(0, n, BATCH_SIZE)):
            batch_size = min(BATCH_SIZE, n - i)
            init_noise = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN)).to(DEVICE)
            outputs = d3pm.sample_with_image_sequence(init_noise, None, stride=40)
            all_tokens.append(outputs[-1].cpu())
    return torch.cat(all_tokens, dim=0)

# -----------------------------------------------------------
# Level 1: Validity check
# -----------------------------------------------------------
def level1_validity(tokens_list):
    print("\n--- Level 1: Validity Check ---")
    valid_tokens = []
    valid_fens = []
    invalid = 0

    for tokens in tokens_list:
        try:
            fen = tokens_to_fen(tokens.numpy())
            board = chess.Board(fen)
            if board.is_valid():
                valid_tokens.append(tokens)
                valid_fens.append(fen)
            else:
                invalid += 1
        except Exception:
            invalid += 1

    total = len(tokens_list)
    valid = len(valid_fens)
    print(f"Valid: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"Invalid: {invalid}/{total} ({100*invalid/total:.1f}%)")
    return valid_fens, valid_tokens

# -----------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------
def extract_features(fen):
    board = chess.Board(fen)
    features = {}

    # piece counts
    for color, color_name in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
        for piece_type, piece_name in [
            (chess.PAWN, 'pawn'), (chess.KNIGHT, 'knight'),
            (chess.BISHOP, 'bishop'), (chess.ROOK, 'rook'),
            (chess.QUEEN, 'queen')
        ]:
            features[f'{color_name}_{piece_name}'] = len(board.pieces(piece_type, color))

    # material balance (white - black in centipawns)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9}
    white_mat = sum(v * len(board.pieces(p, chess.WHITE)) for p, v in values.items())
    black_mat = sum(v * len(board.pieces(p, chess.BLACK)) for p, v in values.items())
    features['material_balance'] = white_mat - black_mat

    # game phase (total material remaining)
    features['total_material'] = white_mat + black_mat

    # pawn structure
    passed_pawns = 0
    doubled_pawns = 0
    isolated_pawns = 0

    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        pawn_files = [chess.square_file(sq) for sq in pawns]

        # doubled pawns
        for f in range(8):
            count = pawn_files.count(f)
            if count > 1:
                doubled_pawns += count - 1

        # isolated pawns
        for f in pawn_files:
            neighbors = [f-1, f+1]
            if not any(n in pawn_files for n in neighbors if 0 <= n <= 7):
                isolated_pawns += 1

        # passed pawns
        enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
        enemy_pawns = board.pieces(chess.PAWN, enemy)
        enemy_files = [chess.square_file(sq) for sq in enemy_pawns]
        for sq in pawns:
            f = chess.square_file(sq)
            if not any(ef in [f-1, f, f+1] for ef in enemy_files):
                passed_pawns += 1

    features['passed_pawns'] = passed_pawns
    features['doubled_pawns'] = doubled_pawns
    features['isolated_pawns'] = isolated_pawns

    # game phase category
    total = features['total_material']
    if total >= 60:
        features['phase'] = 'opening'
    elif total >= 30:
        features['phase'] = 'middlegame'
    else:
        features['phase'] = 'endgame'

    return features

# -----------------------------------------------------------
# Random baseline
# -----------------------------------------------------------
def generate_random_baseline(n=1000):
    print(f"\nGenerating random baseline ({n} positions)...")
    valid_fens = []
    attempts = 0
    max_attempts = n * 200

    while len(valid_fens) < n and attempts < max_attempts:
        tokens = np.random.randint(0, 13, size=72)
        # fix kings
        tokens[:64] = 0
        sq_white = np.random.choice(64, replace=False)
        sq_black = np.random.choice([s for s in range(64) if s != sq_white])
        tokens[sq_white] = 1   # white king
        tokens[sq_black] = 7   # black king
        # scatter some random pieces
        empty_squares = [s for s in range(64) if s != sq_white and s != sq_black]
        n_pieces = np.random.randint(0, 20)
        chosen = np.random.choice(empty_squares, size=min(n_pieces, len(empty_squares)), replace=False)
        for sq in chosen:
            tokens[sq] = np.random.randint(1, 13)
        # metadata
        tokens[64] = np.random.randint(0, 2)
        tokens[65:69] = np.random.randint(0, 2, size=4)
        tokens[69] = np.random.randint(0, 9)
        tokens[70] = np.random.randint(0, 51)
        tokens[71] = np.random.randint(1, 151)

        try:
            fen = tokens_to_fen(tokens)
            board = chess.Board(fen)
            if board.is_valid():
                valid_fens.append(fen)
        except Exception:
            pass
        attempts += 1

    print(f"Random baseline: {len(valid_fens)} valid from {attempts} attempts ({100*len(valid_fens)/attempts:.2f}%)")
    return valid_fens

# -----------------------------------------------------------
# Level 2: Distributional evaluation
# -----------------------------------------------------------
def kl_divergence(p_counts, q_counts, bins):
    p = np.array([p_counts.get(b, 0) for b in bins], dtype=float)
    q = np.array([q_counts.get(b, 0) for b in bins], dtype=float)
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    q = np.where(q == 0, 1e-10, q)
    p = np.where(p == 0, 1e-10, p)
    return float(np.sum(p * np.log(p / q)))

def get_distribution(fens, key, bins):
    counts = defaultdict(int)
    for fen in fens:
        try:
            f = extract_features(fen)
            counts[f[key]] += 1
        except Exception:
            pass
    return counts

def level2_distributional(generated_fens, training_fens, baseline_fens):
    print("\n--- Level 2: Distributional Evaluation ---")
    results = {}

    metrics = [
        ('white_pawn',       list(range(0, 9)),   'White Pawn Count'),
        ('material_balance', list(range(-20, 21)), 'Material Balance'),
        ('passed_pawns',     list(range(0, 10)),   'Passed Pawn Count'),
        ('total_material',   list(range(0, 80)),   'Total Material'),
    ]

    for key, bins, label in metrics:
        gen_dist   = get_distribution(generated_fens, key, bins)
        train_dist = get_distribution(training_fens,  key, bins)
        base_dist  = get_distribution(baseline_fens,  key, bins)

        kl_gen   = kl_divergence(train_dist, gen_dist,  bins)
        kl_base  = kl_divergence(train_dist, base_dist, bins)

        print(f"{label}:")
        print(f"  KL(training || generated):  {kl_gen:.4f}")
        print(f"  KL(training || baseline):   {kl_base:.4f}")
        results[key] = {'kl_generated': kl_gen, 'kl_baseline': kl_base}

        # plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, (dist, title) in zip(axes, [
            (train_dist, 'Training Data'),
            (gen_dist,   'D3PM Generated'),
            (base_dist,  'Random Baseline'),
        ]):
            vals = [dist.get(b, 0) for b in bins]
            total = sum(vals) + 1e-10
            ax.bar(bins, [v/total for v in vals], color='steelblue', alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel(label)
            ax.set_ylabel('Proportion')
        plt.suptitle(label)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{key}_distribution.png", dpi=100)
        plt.close()
        print(f"  Plot saved.")

    return results

# -----------------------------------------------------------
# Game phase distribution
# -----------------------------------------------------------
def phase_distribution(fens, label):
    counts = defaultdict(int)
    for fen in fens:
        try:
            f = extract_features(fen)
            counts[f['phase']] += 1
        except Exception:
            pass
    total = sum(counts.values()) + 1e-10
    print(f"{label}: opening={counts['opening']/total:.1%} "
          f"middlegame={counts['middlegame']/total:.1%} "
          f"endgame={counts['endgame']/total:.1%}")
    return counts

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":

    # load model and generate
    d3pm = load_model()
    generated_tokens = generate_fens(d3pm, n=N_GENERATE)

    # level 1
    valid_fens, valid_tokens = level1_validity(generated_tokens)

    if len(valid_fens) == 0:
        print("No valid FENs generated. Exiting.")
        sys.exit(1)

    # load training sample for comparison
    print("\nLoading training sample for comparison...")
    train_tokens = np.load(TOKENS_PATH)
    sample_idx = np.random.choice(len(train_tokens), size=1000, replace=False)
    training_fens = []
    for idx in sample_idx:
        try:
            fen = tokens_to_fen(train_tokens[idx])
            board = chess.Board(fen)
            if board.is_valid():
                training_fens.append(fen)
        except Exception:
            pass
    print(f"Training sample: {len(training_fens)} FENs")

    # random baseline
    baseline_fens = generate_random_baseline(n=1000)

    # level 2
    results = level2_distributional(valid_fens, training_fens, baseline_fens)

    # game phase
    print("\n--- Game Phase Distribution ---")
    phase_distribution(training_fens, "Training data")
    phase_distribution(valid_fens,    "D3PM generated")
    phase_distribution(baseline_fens, "Random baseline")

    # summary table
    print("\n--- Summary Table ---")
    print(f"{'Metric':<35} {'KL (generated)':<20} {'KL (baseline)':<20}")
    print("-" * 75)
    for key, vals in results.items():
        print(f"{key:<35} {vals['kl_generated']:<20.4f} {vals['kl_baseline']:<20.4f}")

    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print("Evaluation complete.")
