"""
Quick inspection of preprocessed .pt files.
"""

import torch
from pathlib import Path
import sys

PROCESSED_DIR = Path("./processed")

def inspect_pt(pt_path: Path):
    print(f"\n=== {pt_path.name} ===")
    data = torch.load(pt_path, map_location="cpu")
    
    print("Keys:", list(data.keys()))
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v}")
    
    # Show a few stats on the Cp values
    if "cp" in data:
        cp = data["cp"]
        print(f"\n  Cp stats: min={cp.min():.4f}, max={cp.max():.4f}, mean={cp.mean():.4f}")

if __name__ == "__main__":
    pt_files = sorted(PROCESSED_DIR.glob("*.pt"))
    
    if not pt_files:
        print(f"No .pt files found in {PROCESSED_DIR.resolve()}")
        sys.exit(1)
    
    print(f"Found {len(pt_files)} processed files.")
    
    # Inspect the first one in detail
    inspect_pt(pt_files[0])
    
    # Optionally inspect all
    if len(pt_files) > 1:
        print("\n--- Summary for all files ---")
        for f in pt_files:
            data = torch.load(f, map_location="cpu")
            print(f"{f.name}: n_points={data['n_points']}, cp_range=[{data['cp'].min():.3f}, {data['cp'].max():.3f}]")