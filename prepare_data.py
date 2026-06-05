import pandas as pd
import numpy as np
import chess
from tokenizer import fen_to_tokens
from tqdm import tqdm

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
CSV_PATH    = '/project/lichess_db_puzzle.csv'
OUTPUT_PATH = '/project/tokens.npy'
NUM_SAMPLES = 500_000  # start with 500k, can increase later

# -----------------------------------------------------------
# Load FENs
# -----------------------------------------------------------
print(f"Loading {NUM_SAMPLES} FENs from CSV...")
df = pd.read_csv(CSV_PATH, usecols=['FEN'], nrows=NUM_SAMPLES)
fens = df['FEN'].tolist()
print(f"Loaded {len(fens)} FENs")

# -----------------------------------------------------------
# Tokenize
# -----------------------------------------------------------
print("Tokenizing...")
tokens_list = []
skipped = 0

for fen in tqdm(fens):
    tokens = fen_to_tokens(fen)
    if tokens is not None:
        tokens_list.append(tokens)
    else:
        skipped += 1

print(f"Tokenized: {len(tokens_list)} | Skipped: {skipped}")

# -----------------------------------------------------------
# Save
# -----------------------------------------------------------
tokens_array = np.stack(tokens_list, axis=0)
np.save(OUTPUT_PATH, tokens_array)
print(f"Saved to {OUTPUT_PATH}")
print(f"Array shape: {tokens_array.shape}")
print(f"dtype: {tokens_array.dtype}")
