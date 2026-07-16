"""
Preprocessing pipeline for DrivAerML boundary_i.vtp files.
- Load with PyVista
- Triangulate if needed
- Decimate to target number of faces (~5k for practicality)
- Transfer CpMeanTrim from original mesh using nearest-neighbor lookup on cell centers
- Extract: cell centers or vertices + normals + CpMeanTrim
- Local normalize coords (zero mean, unit std per axis) + store stats
- Save as dict in .pt for fast loading
"""

import pyvista as pv
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy.spatial import cKDTree

def process_vtp(vtp_path: Path, target_faces: int = 5000, use_cell_centers: bool = True):
    print(f"  [1/7] Reading mesh: {vtp_path.name}")
    mesh = pv.read(str(vtp_path))
    print(f"        -> Original: {mesh.n_points} points, {mesh.n_faces} faces")

    # Check for CpMeanTrim early
    cp = mesh.cell_data.get("CpMeanTrim", mesh.point_data.get("CpMeanTrim", None))
    if cp is None:
        raise IOError("        -> WARNING: CpMeanTrim not found in original mesh.")

    # Ensure triangulated surface
    print(f"  [2/7] Checking triangulation (is_all_triangles={mesh.is_all_triangles})")
    if not mesh.is_all_triangles:
        print("        -> Triangulating...")
        mesh = mesh.triangulate()
        print(f"        -> After triangulation: {mesh.n_faces} faces")
    else:
        print("        -> Already triangulated, skipping.")

    # Decimate + transfer CpMeanTrim via nearest-neighbor lookup on cell centers
    print(f"  [3/7] Decimation check (target_faces={target_faces})")
    if mesh.n_faces > target_faces:
        reduction = 1 - target_faces / mesh.n_faces
        print(f"        -> Decimating with target_reduction={reduction:.3f}")
        decimated = mesh.decimate(target_reduction=reduction, volume_preservation=True)
        print(f"        -> After decimation: {decimated.n_faces} faces")

        # Transfer CpMeanTrim using nearest cell center lookup (more reliable than interpolate)
        print("        -> Transferring CpMeanTrim via nearest-neighbor lookup on cell centers...")
        if 'CpMeanTrim' in mesh.cell_data:
            orig_centers = mesh.cell_centers().points
            dec_centers = decimated.cell_centers().points
            tree = cKDTree(orig_centers)
            _, idx = tree.query(dec_centers, k=1)
            decimated.cell_data['CpMeanTrim'] = mesh.cell_data['CpMeanTrim'][idx]
            print("        -> CpMeanTrim transferred successfully (cell data).")
        else:
            print("        -> WARNING: CpMeanTrim not found as cell_data. Trying point_data...")
            if 'CpMeanTrim' in mesh.point_data:
                # Fallback: interpolate from point data if needed
                decimated = decimated.interpolate(mesh)
                print("        -> CpMeanTrim transferred via point_data interpolation.")

        mesh = decimated
        print(f"        -> Final mesh after transfer: {mesh.n_points} points, {mesh.n_faces} faces")
    else:
        print("        -> No decimation needed.")

    # Verify CpMeanTrim survived
    cp = mesh.cell_data.get("CpMeanTrim", mesh.point_data.get("CpMeanTrim", None))
    if cp is None:
        raise IOError("        -> WARNING: CpMeanTrim lost after processing. Using zeros.")
        cp = np.zeros(mesh.n_cells if use_cell_centers else mesh.n_points)
    else:
        print("        -> CpMeanTrim successfully preserved.")

    # Get points and normals
    print(f"  [4/7] Extracting centers/normals (use_cell_centers={use_cell_centers})")
    if use_cell_centers:
        centers = mesh.cell_centers().points
        mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
        normals = mesh.cell_data['Normals'] if 'Normals' in mesh.cell_data else np.zeros_like(centers)
        cp = mesh.cell_data.get("CpMeanTrim", mesh.point_data.get("CpMeanTrim", np.zeros(len(centers))))
    else:
        centers = mesh.points
        mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        normals = mesh.point_data['Normals']
        cp = mesh.point_data.get("CpMeanTrim", mesh.cell_data.get("CpMeanTrim", np.zeros(len(centers))))

    print(f"        -> Extracted {len(centers)} centers, Cp shape: {cp.shape}")

    # Local normalization
    print(f"  [5/7] Normalizing coordinates")
    coords_mean = centers.mean(axis=0)
    coords_std = centers.std(axis=0) + 1e-8
    coords_norm = (centers - coords_mean) / coords_std
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    print(f"        -> coords_mean={coords_mean}, coords_std={coords_std}")

    # Build data dict
    print(f"  [6/7] Building data dictionary and converting to torch tensors")
    data = {
        'coords': torch.tensor(coords_norm, dtype=torch.float32),
        'normals': torch.tensor(normals, dtype=torch.float32),
        'cp': torch.tensor(cp, dtype=torch.float32).unsqueeze(-1),
        'coords_mean': torch.tensor(coords_mean, dtype=torch.float32),
        'coords_std': torch.tensor(coords_std, dtype=torch.float32),
        'original_points': torch.tensor(centers, dtype=torch.float32),
        'n_points': len(centers),
        'source_file': str(vtp_path)
    }
    data['features'] = torch.cat([data['coords'], data['normals']], dim=1)
    print(f"  [7/7] Done processing {vtp_path.name}")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../1_Data_Processing/references/data')
    parser.add_argument('--out_dir', type=str, default='../1_Data_Processing/output/processed')
    parser.add_argument('--target_faces', type=int, default=5000)
    parser.add_argument('--train_runs', type=str, default='80,98,102,208,213,281,397,425,445,474')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_runs = [f'run_{r.strip()}' for r in args.train_runs.split(',')]
    print(f"Train runs to process: {train_runs}\n")
    
    for run in tqdm(train_runs, desc='Processing train runs'):
        vtp = Path(args.data_dir) / run / f'boundary_{run.split("_")[1]}.vtp'
        print(f"\n=== Processing {run} ===")
        print(f"Looking for: {vtp}")
        if not vtp.exists():
            print(f"Warning: {vtp} not found. Skipping.")
            continue
        data = process_vtp(vtp, target_faces=args.target_faces)
        torch.save(data, out_dir / f'{run}.pt')
        print(f"Saved {run}: {data['n_points']} points -> {out_dir / f'{run}.pt'}")

    print("\nPreprocessing complete. Now run train.py")

if __name__ == '__main__':
    main()