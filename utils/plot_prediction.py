"""
plot_prediction.py
Interactive 3D viewer for inference results.
Shows predicted Cp, ground truth (if available), absolute error, relative error,
and directional correctness.
"""

import pyvista as pv
import numpy as np
from pathlib import Path
import argparse

def plot_prediction(vtp_path: str, show_relative: bool = True):
    mesh = pv.read(vtp_path)

    print(f"Loaded: {vtp_path}")
    print("Available arrays:", mesh.cell_data.keys())

    if 'pred_Cp' in mesh.cell_data and 'GT_Cp' in mesh.cell_data:
        pred = mesh.cell_data['pred_Cp']
        gt = mesh.cell_data['GT_Cp']

        # Absolute error (log scale)
        abs_error = np.abs(pred - gt) + 1e-12
        mesh.cell_data['abs_error'] = np.log10(abs_error)

        # Relative error (log scale)
        rel_error = np.abs(pred - gt) / (np.abs(gt) + 1e-12)
        mesh.cell_data['rel_error'] = np.log10(rel_error + 1e-12)

        # Directional correctness
        direction_correct = (np.sign(pred) == np.sign(gt)).astype(float)
        mesh.cell_data['direction_correct'] = direction_correct

        print(f"Directional correctness: {direction_correct.mean() * 100:.1f}%")

    # Define which scalars to show
    scalars_to_show = []
    if 'pred_Cp' in mesh.cell_data:
        scalars_to_show.append('pred_Cp')
    if 'GT_Cp' in mesh.cell_data:
        scalars_to_show.append('GT_Cp')
    if 'abs_error' in mesh.cell_data:
        scalars_to_show.append('abs_error')
    if 'rel_error' in mesh.cell_data and show_relative:
        scalars_to_show.append('rel_error')
    if 'direction_correct' in mesh.cell_data:
        scalars_to_show.append('direction_correct')

    if not scalars_to_show:
        print("No prediction arrays found!")
        return

    # Create multi-view plot with dark background
    n = len(scalars_to_show)
    plotter = pv.Plotter(shape=(1, n), window_size=(300 * n, 800))
    plotter.set_background("#1e1e1e")   # Dark background

    for i, name in enumerate(scalars_to_show):
        plotter.subplot(0, i)

        if name == 'direction_correct':
            clim = [0, 1]
            cmap = "RdYlGn"
            title = "Direction Correct\n(1=correct sign)"
        elif name in ['abs_error', 'rel_error']:
            clim = (-2, 0)
            cmap = "magma"
            title = name
        else:
            clim = [-1.2, 1.01]
            cmap = "coolwarm"
            title = name

        plotter.add_mesh(
            mesh,
            scalars=name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args={"title": title, "color": "white"}
        )
        plotter.add_text(name, position="upper_edge", font_size=12, color="white")

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vtp', type=str, required=True,
                        help='Path to the predicted .vtp file')
    parser.add_argument('--no_rel', action='store_true',
                        help='Do not show relative error')
    args = parser.parse_args()

    plot_prediction(args.vtp, show_relative=not args.no_rel)