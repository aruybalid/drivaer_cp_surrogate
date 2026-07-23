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
from model_factory import MODEL_REGISTRY, MODELS_WITH_PARAMS, get_model

class DrivAerDataset(Dataset):
    def __init__(self, processed_dir, run_list, augment=True):
        self.files = [Path(processed_dir) / f'{r}.pt' for r in run_list]
        self.augment = augment
        self.data = [torch.load(f) for f in self.files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        feats = d['features'].clone()  # [N, 6]
        cp = d['cp'].clone()           # [N, 1]

        # Load the 16 geometric parameters if available
        if 'params' in d:
            params = d['params'].clone()   # shape: (16,)
        else:
            params = torch.zeros(16)       # fallback for old processed files

        if self.augment:
            # Jitter coords slightly (simulate mesh variation)
            feats[:, :3] += torch.randn_like(feats[:, :3]) * 0.02
            # Jitter normals
            feats[:, 3:] += torch.randn_like(feats[:, 3:]) * 0.05
            feats[:, 3:] = feats[:, 3:] / (feats[:, 3:].norm(dim=1, keepdim=True) + 1e-8)
            # Random point dropout (keep 80-100%)
            if torch.rand(1) > 0.5:
                keep = int(feats.shape[0] * np.random.uniform(0.8, 1.0))
                perm = torch.randperm(feats.shape[0])[:keep]
                feats = feats[perm]
                cp = cp[perm]

        return feats, cp, params

def collate_fn(batch):
    max_n = max(b[0].shape[0] for b in batch)
    feats_padded, cp_padded, masks, params = [], [], [], []

    for f, c, p in batch:
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
        params.append(p)

    return (
        torch.stack(feats_padded),
        torch.stack(cp_padded),
        torch.stack(masks),
        torch.stack(params)          # [B, 16]
    )

def train_one_epoch(model, loader, optimizer, criterion, device, model_name):
    model.train()
    total_loss = 0

    for batch in loader:
        if len(batch) == 4:
            feats, cp, masks, params = batch
            params = params.to(device)
        else:
            feats, cp, masks = batch
            params = None

        feats, cp, masks = feats.to(device), cp.to(device), masks.to(device)

        # Only pass params if the model actually needs them
        if model_name in MODELS_WITH_PARAMS:
            pred = model(feats, params)
        else:
            pred = model(feats)

        loss = criterion(pred * masks.unsqueeze(-1), cp * masks.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, model_name):
    model.eval()
    total_loss = 0

    for batch in loader:
        if len(batch) == 4:
            feats, cp, masks, params = batch
            params = params.to(device)
        else:
            feats, cp, masks = batch
            params = None

        feats, cp, masks = feats.to(device), cp.to(device), masks.to(device)

        if model_name in MODELS_WITH_PARAMS:
            pred = model(feats, params)
        else:
            pred = model(feats)

        loss = criterion(pred * masks.unsqueeze(-1), cp * masks.unsqueeze(-1))
        total_loss += loss.item() * feats.size(0)

    return total_loss / len(loader.dataset)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train DrivAer Cp Surrogate Models")
    
    # Model selection
    parser.add_argument('--model', type=str, default='pointnet_with_params',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to train')
    
    # Training parameters
    parser.add_argument('--processed_dir', type=str, default='./references/processed')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden size: {args.hidden}")

    # all_runs = ['run_80','run_98','run_102','run_208','run_213','run_281','run_397','run_425','run_445','run_474']
    all_runs = ['run_80','run_98','run_102','run_213','run_281','run_397','run_425','run_445','run_474'] # exclude run_208 from all runs due to extreme outliers
    train_runs = all_runs[:8]
    val_runs = all_runs[8:]

    print(f"\nTrain runs ({len(train_runs)}): {train_runs}")
    print(f"Val runs   ({len(val_runs)}): {val_runs}")

    print("\n[1/4] Loading datasets...")
    train_ds = DrivAerDataset(args.processed_dir, train_runs, augment=True)
    val_ds = DrivAerDataset(args.processed_dir, val_runs, augment=False)
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    sample_feats, sample_cp, sample_params = train_ds[0]
    print(f"\n[2/4] Sample inspection:")
    print(f"  features shape: {sample_feats.shape}")
    print(f"  cp shape:       {sample_cp.shape}")
    print(f"  params shape:   {sample_params.shape}")

    print("\n[3/4] Creating dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("\n[4/4] Building model...")
    
    # Use the model factory
    model = get_model(args.model, hidden=args.hidden).to(device)
    print(f"  Created model: {args.model}")

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

        # train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # val_loss = eval_one_epoch(model, val_loader, criterion, device)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.model)
        val_loss   = eval_one_epoch(model, val_loader, criterion, device, args.model)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={epoch_time:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'model_name': args.model
            }, './output/best_model.pt')
            print("  ✓ Saved new best model")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    Path('./output').mkdir(exist_ok=True)
    main()