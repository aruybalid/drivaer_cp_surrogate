"""
Inference script + reusable functions.
Load trained model, preprocess new vtp identically, predict Cp, attach to mesh, export new vtp or return data.
Use for web app backend and held-out testing.
"""

import torch
import pyvista as pv
import numpy as np
from pathlib import Path
from preprocess import process_vtp
from model import PointNetCpRegressor
from scipy.spatial import cKDTree
import argparse

def predict_cp(vtp_path: str,
               model_path: str = '../2_Surrogate_Building/22_Surrogate_Testing/references/best_model.pt',
               target_faces: int = 5000,
               device='cpu'):
    """
    Run inference on a .vtp file.

    Parameters
    ----------
    vtp_path : str
        Path to the input .vtp file.
    model_path : str
        Path to the trained model checkpoint.
    target_faces : int
        Target number of faces if decimation is enabled.
    device : str
        'cpu' or 'cuda'.
    """
    device = torch.device(device)

    # Preprocess exactly as during training (data now in torch pt format)
    data = process_vtp(Path(vtp_path), target_faces=target_faces, use_cell_centers=True)

    # Load model
    model = PointNetCpRegressor(input_dim=6, hidden=128)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        feats = data['features'].unsqueeze(0).to(device)  # [1, N, 6]
        pred = model(feats).squeeze(0).cpu().numpy()      # [N, 1]
    print("Prediction complete.")

    # 3. Build output mesh from the already processed data
    points = data['original_points'].numpy()
    mesh = pv.PolyData(points)

    # Attach prediction
    mesh.cell_data['pred_Cp'] = pred.flatten()

    # Add ground truth and error if available
    if 'cp' in data:
        mesh.cell_data['GT_Cp']   = data['cp'].squeeze().numpy().flatten()
        mesh.cell_data['error']   = np.abs(pred.flatten() - mesh.cell_data['GT_Cp'])
    else:
        print("Warning: Ground truth CpMeanTrim not found in the mesh. Only predictions will be available.")
    return mesh, data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vtp_path', type=str, required=True,
                        help='Path to the input .vtp file')
    parser.add_argument('--model', type=str, default='./references/best_model.pt',
                        help='Path to the trained model')
    parser.add_argument('--out', type=str, default='./output/predicted.vtp',
                        help='Output file name')
    parser.add_argument('--target_faces', type=int, default=5000,
                        help='Target number of faces. If chosen to be larger than the number of data points), decimation won\'t be activated.')
    args = parser.parse_args()

    mesh, _ = predict_cp(args.vtp_path,
                         model_path=args.model,
                         target_faces=args.target_faces)

    mesh.save(args.out)
    print(f"Saved predicted mesh to {args.out}")

    if 'pred_Cp' in mesh.cell_data:
        pred = mesh.cell_data['pred_Cp']
        print(f"Pred Cp  → mean: {pred.mean():.4f}, range: [{pred.min():.4f}, {pred.max():.4f}]")

    if 'error' in mesh.cell_data:
        err = mesh.cell_data['error']
        print(f"Error    → mean: {err.mean():.4f}, max: {err.max():.4f}")


if __name__ == '__main__':
    main()