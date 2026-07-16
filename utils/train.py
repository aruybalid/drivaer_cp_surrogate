"""
Training script for PointNet Cp regressor on 10 DrivAerML cases.
EXTREMELY VERBOSE version to diagnose hangs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import time
from tqdm import tqdm
from model import PointNetCpRegressor

# ============================================================
# Dataset
# ============================================================
class DrivAerDataset(Dataset):
    def __init__(self, processed_dir, run_list, augment=True):
        self.files = [Path(processed_dir) / f'{r}.pt' for r in run_list]
        self.augment = augment
        self.data = [torch.load(f) for f in self.files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        feats = d['features'].clone()
        cp = d['cp'].clone()

        if self.augment:
            feats[:, :3] += torch.randn_like(feats[:, :3]) * 0.02
            feats[:, 3:] += torch.randn_like(feats[:, 3:]) * 0.05
            feats[:, 3:] = feats[:, 3:] / (feats[:, 3:].norm(dim=1, keepdim=True) + 1e-8)
            if torch.rand(1) > 0.5:
                keep = int(feats.shape[0] * np.random.uniform(0.8, 1.0))
                perm = torch.randperm(feats.shape[0])[:keep]
                feats = feats[perm]
                cp = cp[perm]

        return feats, cp

# ============================================================
# VERY VERBOSE Collate function
# ============================================================
def collate_fn(batch):

    # Print point counts of every sample in this batch
    ns = [b[0].shape[0] for b in batch]

    max_n = max(ns)

    feats_padded, cp_padded, masks = [], [], []

    for i, (f, c) in enumerate(batch):
        n = f.shape[0]
        pad_len = max_n - n

        if pad_len > 0:
            f = torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)
            c = torch.cat([c, torch.zeros(pad_len, 1)], dim=0)
            mask = torch.cat([torch.ones(n), torch.zeros(pad_len)])
        else:
            mask = torch.ones(n)

        feats_padded.append(f)
        cp_padded.append(c)
        masks.append(mask)

    stacked_feats = torch.stack(feats_padded)

    return stacked_feats, torch.stack(cp_padded), torch.stack(masks)

# ============================================================
# Training / Evaluation loops
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    print("[train_one_epoch] Starting epoch training loop...")

    pbar = tqdm(loader, desc="  Training batches", leave=False)

    for batch_idx, (feats, cp, masks) in enumerate(pbar):
        feats, cp, masks = feats.to(device), cp.to(device), masks.to(device)
        pred = model(feats)
        loss = criterion(pred * masks.unsqueeze(-1), cp * masks.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="  Validation batches", leave=False)
    for feats, cp, masks in pbar:
        feats, cp, masks = feats.to(device), cp.to(device), masks.to(device)
        pred = model(feats)
        loss = criterion(pred * masks.unsqueeze(-1), cp * masks.unsqueeze(-1))
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', type=str, default='./references/processed')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    all_runs = ['run_80','run_98','run_102','run_208','run_213','run_281','run_397','run_425','run_445','run_474']
    train_runs = all_runs[:8]
    val_runs = all_runs[8:]

    print(f"\nTrain runs ({len(train_runs)}): {train_runs}")
    print(f"Val runs   ({len(val_runs)}): {val_runs}")

    print("\n[1/4] Loading datasets...")
    train_ds = DrivAerDataset(args.processed_dir, train_runs, augment=True)
    val_ds = DrivAerDataset(args.processed_dir, val_runs, augment=False)
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    sample_feats, sample_cp = train_ds[0]
    print(f"\n[2/4] Sample inspection:")
    print(f"  features shape: {sample_feats.shape}")
    print(f"  cp shape:       {sample_cp.shape}")

    print("\n[3/4] Creating dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("\n[4/4] Building model...")
    model = PointNetCpRegressor(input_dim=6, hidden=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val = float('inf')
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, './output/best_model.pt')
            print("  ✓ Saved new best model")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    Path('./output').mkdir(exist_ok=True)
    main()