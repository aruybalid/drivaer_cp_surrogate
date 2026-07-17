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
from scipy.spatial import cKDTree
import argparse
from model_factory import MODEL_REGISTRY, MODELS_WITH_PARAMS , get_model  # or define inline

def get_model(model_name, input_dim=6, num_params=16, hidden=128):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    if model_name in MODELS_WITH_PARAMS:
        return model_class(input_dim=input_dim, num_params=num_params, hidden=hidden)
    else:
        return model_class(input_dim=input_dim, hidden=hidden)



def predict_cp(vtp_path, model_path, target_faces=5000, model_name='pointnet', params=None, device='cpu'):
    device = torch.device(device)
    data = process_vtp(Path(vtp_path), target_faces=target_faces, use_cell_centers=True)

    # Create the correct model based on the dropdown selection
    model = get_model(model_name).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        feats = data['features'].unsqueeze(0).to(device)

        # Auto-load params from the processed data if available
        if params is None and 'params' in data:
            params = data['params'].unsqueeze(0).to(device)

        # Call the model correctly depending on whether it needs params
        print("\nRunning inference...")
        if model_name in MODELS_WITH_PARAMS:
            if params is not None:
                pred = model(feats, params)
            else:
                # Fallback: create dummy params if none available
                params = torch.zeros(1, 16, device=device)
                pred = model(feats, params)
        else:
            pred = model(feats)

        pred = pred.squeeze(0).cpu().numpy().flatten()
        print("Prediction complete.")
        
    # Build output mesh from the processed data
    points = data['original_points'].numpy()
    mesh = pv.PolyData(points)
    mesh.cell_data['pred_Cp'] = pred

    if 'cp' in data:
        gt = data['cp'].squeeze().numpy()
        mesh.cell_data['GT_Cp'] = gt
        mesh.cell_data['error'] = np.abs(pred - gt)

    return mesh, data


def main():
    parser = argparse.ArgumentParser(description="Run DrivAer Cp Inference")
    
    # Model selection
    parser.add_argument('--model', type=str, default='pointnet_with_params',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture used during training')
    
    # Input / Output
    parser.add_argument('--vtp_path', type=str, required=True,
                        help='Path to the input .vtp file')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--out', type=str, default='predicted.vtp',
                        help='Output file name for the predicted mesh')
    
    # Processing options
    parser.add_argument('--target_faces', type=int, default=15000,
                        help='Target number of faces after decimation')
    
    args = parser.parse_args()

    print("=" * 60)
    print("INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input VTP: {args.vtp_path}")
    print(f"Model path: {args.model_path}")
    print(f"Target faces: {args.target_faces}")
    print(f"Output: {args.out}")

    # Run inference
    print("\nRunning inference...")
    mesh, data = predict_cp(
        vtp_path=args.vtp_path,
        model_path=args.model_path,
        target_faces=args.target_faces,
        model_name=args.model
    )

    # Save result
    mesh.save(args.out)
    print(f"\nSaved predicted mesh to: {args.out}")

    # Print basic statistics
    if 'pred_Cp' in mesh.cell_data:
        pred = mesh.cell_data['pred_Cp']
        print(f"\nPrediction Statistics:")
        print(f"  Mean Cp: {pred.mean():.4f}")
        print(f"  Cp Range: [{pred.min():.4f}, {pred.max():.4f}]")

    if 'error' in mesh.cell_data:
        error = mesh.cell_data['error']
        print(f"\nError Statistics:")
        print(f"  Mean Absolute Error: {error.mean():.4f}")
        print(f"  Max Absolute Error:  {error.max():.4f}")

    print("\nInference complete.")

if __name__ == '__main__':
    main()