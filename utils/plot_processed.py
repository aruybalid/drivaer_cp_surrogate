"""
plot_processed.py
Interactive 3D viewer for the preprocessed (decimated) point clouds.

Usage:
    python plot_processed.py                # all runs
    python plot_processed.py --dataset 98   # single run
"""

import torch
import pyvista as pv
from pathlib import Path
import argparse

PROCESSED_DIR = Path("./references/processed")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=None,
                    help="Run number (e.g. 98). If omitted, load all processed runs.")
args = parser.parse_args()

def find_processed_files(dataset_num=None):
    if dataset_num is not None:
        pt = PROCESSED_DIR / f"run_{dataset_num}.pt"
        if not pt.exists():
            raise FileNotFoundError(f"{pt} not found")
        return [pt]
    else:
        return sorted(PROCESSED_DIR.glob("run_*.pt"))

pt_files = find_processed_files(args.dataset)
print(f"Found {len(pt_files)} processed files.")

# ============================================================
# Interactive viewer (Dark Theme)
# ============================================================
def interactive_viewer(files):
    plotter = pv.Plotter()
    plotter.set_background("#1e1e1e")          # Dark background
    current_idx = [0]

    def load_cloud(idx):
        plotter.clear()
        data = torch.load(files[idx], map_location="cpu")
        points = data["original_points"].numpy()
        cp = data["cp"].squeeze().numpy()

        cloud = pv.PolyData(points)
        cloud["CpMeanTrim"] = cp

        plotter.add_mesh(
            cloud,
            scalars="CpMeanTrim",
            cmap="coolwarm",
            render_points_as_spheres=False,
            point_size=4,
            show_scalar_bar=True,
            clim=[-2.5, 1.01],
            scalar_bar_args={"title": "CpMeanTrim", "color": "white"}
        )
        plotter.add_text(f"{files[idx].name}  ({idx+1}/{len(files)})", 
                         position="upper_edge", font_size=12, color="white")
        plotter.render()

    def next_cloud():
        current_idx[0] = (current_idx[0] + 1) % len(files)
        load_cloud(current_idx[0])

    def prev_cloud():
        current_idx[0] = (current_idx[0] - 1) % len(files)
        load_cloud(current_idx[0])

    plotter.add_key_event("n", next_cloud)
    plotter.add_key_event("p", prev_cloud)
    plotter.add_key_event("q", lambda: plotter.close())

    load_cloud(0)
    print("\nControls: n = next, p = previous, q = quit")
    plotter.show()

interactive_viewer(pt_files)