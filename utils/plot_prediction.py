"""
plot_prediction.py
Interactive 3D viewer for inference results.
Shows predicted Cp, ground truth (if available), absolute error, and relative error.
"""

import pyvista as pv
import numpy as np
from pathlib import Path
import argparse

def plot_prediction(vtp_path: str, show_relative: bool = True):
    mesh = pv.read(vtp_path)

    print(f"Loaded: {vtp_path}")
    print("Available arrays:", mesh.cell_data.keys())

    # Compute relative error if possible
    if 'pred_Cp' in mesh.cell_data and 'GT_Cp' in mesh.cell_data:
        pred = mesh.cell_data['pred_Cp']
        gt = mesh.cell_data['GT_Cp']
        rel_error = np.abs(pred - gt) / (np.abs(gt) + 1e-8)
        mesh.cell_data['rel_error'] = rel_error
        print(f"Relative error computed. Mean = {rel_error.mean():.4f}")

    # Choose which scalars to show
    scalars_to_show = []
    if 'pred_Cp' in mesh.cell_data:
        scalars_to_show.append('pred_Cp')
    if 'GT_Cp' in mesh.cell_data:
        scalars_to_show.append('GT_Cp')
    if 'error' in mesh.cell_data:
        scalars_to_show.append('error')
    if 'rel_error' in mesh.cell_data and show_relative:
        scalars_to_show.append('rel_error')

    if not scalars_to_show:
        print("No prediction arrays found!")
        return

    # Interactive multi-view plot
    n = len(scalars_to_show)
    plotter = pv.Plotter(shape=(1, n), window_size=(300 * n, 800))

    for i, name in enumerate(scalars_to_show):
        plotter.subplot(0, i)
        clim = [-1.2, 1.2]
        if name in ['error', 'rel_error']:
            clim = (0, mesh.cell_data[name].max())
        cmap = "coolwarm" if name in ['pred_Cp', 'GT_Cp'] else "magma"

        plotter.add_mesh(
            mesh,
            scalars=name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args={"title": name}
        )
        plotter.add_text(name, position="upper_edge", font_size=12)

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vtp', type=str, required=True,
                        help='Path to the predicted .vtp file (e.g. ./output/predicted_run_1.vtp)')
    parser.add_argument('--no_rel', action='store_true',
                        help='Do not compute/show relative error')
    args = parser.parse_args()

    plot_prediction(args.vtp, show_relative=not args.no_rel)