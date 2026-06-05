import sys
sys.path.insert(0, '/project/d3pm')

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm
import os

from d3pm_runner import D3PM
from dit import DDiT_Llama

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
TOKENS_PATH     = '/project/tokens.npy'
CHECKPOINT_DIR  = '/project/checkpoints'
SEQ_LEN         = 72
VOCAB_SIZE      = 151
BATCH_SIZE      = 256
N_EPOCHS        = 10
N_T             = 1000
LR              = 2e-4
WARMUP_STEPS    = 100
SAMPLE_EVERY    = 500
SAVE_EVERY      = 2000
N_SAMPLES       = 16
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------
class FENDataset(Dataset):
    def __init__(self, path):
        self.tokens = np.load(path)
        print(f"Loaded dataset: {self.tokens.shape}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long)

# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate_samples(samples, step):
    import chess
    from tokenizer import tokens_to_fen

    valid = 0
    total = len(samples)
    for token_seq in samples:
        try:
            fen = tokens_to_fen(token_seq.cpu().numpy())
            board = chess.Board(fen)
            if board.is_valid():
                valid += 1
        except Exception:
            pass

    pct = 100.0 * valid / total
    print(f"\n[Step {step}] Valid FENs: {valid}/{total} ({pct:.1f}%)\n")
    return pct

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":

    print(f"Device: {DEVICE}")

    dataset = FENDataset(TOKENS_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = DDiT_Llama(VOCAB_SIZE, dim=512, n_layers=6)
    d3pm = D3PM(
        model,
        n_T=N_T,
        num_classes=VOCAB_SIZE,
        hybrid_loss_coeff=0.01
    ).to(DEVICE)

    total_params = sum(p.numel() for p in d3pm.x0_model.parameters())
    print(f"Model parameters: {total_params:,}")

    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=LR)
    total_steps = N_EPOCHS * math.ceil(len(dataloader))
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optim,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    global_step = 0
    loss_ema = None

    for epoch in range(N_EPOCHS):
        d3pm.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")

        for x in pbar:
            optim.zero_grad()
            x = x.to(DEVICE)

            loss, info = d3pm(x)
            loss.backward()

            norm = torch.nn.utils.clip_grad_norm_(
                d3pm.x0_model.parameters(), 5.0
            )

            optim.step()
            lr_scheduler.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            pbar.set_description(
                f"Epoch {epoch+1} | loss: {loss_ema:.4f} "
                f"| vb: {info['vb_loss']:.4f} "
                f"| ce: {info['ce_loss']:.4f} "
                f"| norm: {norm:.4f}"
            )

            global_step += 1

            if global_step % SAMPLE_EVERY == 1:
                d3pm.eval()
                with torch.no_grad():
                    init_noise = torch.randint(
                        0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN)
                    ).to(DEVICE)
                    outputs = d3pm.sample_with_image_sequence(
                        init_noise, None, stride=40
                    )
                    samples = outputs[-1]
                evaluate_samples(samples, global_step)
                d3pm.train()

            if global_step % SAVE_EVERY == 1:
                ckpt_path = f"{CHECKPOINT_DIR}/d3pm_fen_step{global_step}.pth"
                torch.save({
                    'step': global_step,
                    'model_state_dict': d3pm.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss_ema': loss_ema,
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    torch.save({
        'step': global_step,
        'model_state_dict': d3pm.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss_ema': loss_ema,
    }, f"{CHECKPOINT_DIR}/d3pm_fen_final.pth")
    print("Training complete. Final model saved.")
