"""
plot_data.py
Interactive 3D viewer + geo-parameter analysis for DrivAer boundary vtp files.

Data is now organized as:
    data/
        run_80/
            boundary_80.vtp
            geo_parameters_80.csv
        run_98/
            ...
"""

import pyvista as pv
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Apply dark theme to all matplotlib plots
plt.style.use('dark_background')

# ============================================================
# Argument parser
# ============================================================
parser = argparse.ArgumentParser(description="Plot DrivAer boundary data")
parser.add_argument("--data_dir", type=str, default="../1_Data_Processing/references/data",
                    help="Path to the data directory containing run_XX/ folders.")
parser.add_argument("--dataset", type=int, default=None,
                    help="Dataset number (e.g. 98 for run_98/boundary_98.vtp). If omitted, all datasets are loaded.")
parser.add_argument("--global_plots", action="store_true", default=False,
                    help="If provided, generate overview line plots of global vehicle info across the entire data-set.")

args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

# ============================================================
# 1. Find boundary vtp files (single or all)
# ============================================================
def find_vtp_files(dataset_num=None):
    if dataset_num is not None:
        run_dir = DATA_DIR / f"run_{dataset_num}"
        vtp = run_dir / f"boundary_{dataset_num}.vtp"
        if not vtp.exists():
            raise FileNotFoundError(f"{vtp} not found")
        print(f"Single dataset mode: {vtp}")
        return [vtp]
    else:
        # Find all run_*/boundary_*.vtp
        files = sorted(DATA_DIR.glob("run_*/boundary_*.vtp"))
        print(f"Found {len(files)} boundary files (all datasets).")
        return files

vtp_files = find_vtp_files(args.dataset)

# ============================================================
# 2. Interactive 3D viewer (cycle through files) - Dark Theme
# ============================================================
def interactive_viewer(files):
    plotter = pv.Plotter()
    plotter.set_background("#1e1e1e")          # Dark background
    current_idx = [0]

    def load_mesh(idx):
        plotter.clear()
        mesh = pv.read(str(files[idx]))
        cp = mesh.cell_data.get("CpMeanTrim", mesh.point_data.get("CpMeanTrim", None))

        if cp is not None:
            plotter.add_mesh(
                mesh,
                scalars="CpMeanTrim",
                cmap="coolwarm",
                show_scalar_bar=True,
                clim=[-2.5, 1.01],
                scalar_bar_args={"title": "CpMeanTrim", "color": "white"}
            )
        else:
            plotter.add_mesh(mesh, color="lightgray")

        plotter.add_text(f"{files[idx].name}  ({idx+1}/{len(files)})", 
                         position="upper_edge", font_size=12, color="white")
        plotter.render()

    def next_mesh():
        current_idx[0] = (current_idx[0] + 1) % len(files)
        load_mesh(current_idx[0])

    def prev_mesh():
        current_idx[0] = (current_idx[0] - 1) % len(files)
        load_mesh(current_idx[0])

    plotter.add_key_event("n", next_mesh)
    plotter.add_key_event("p", prev_mesh)
    plotter.add_key_event("q", lambda: plotter.close())

    load_mesh(0)
    print("\nInteractive viewer controls:")
    print("  n = next mesh")
    print("  p = previous mesh")
    print("  q = quit")
    plotter.show()

# Run interactive viewer
interactive_viewer(vtp_files)

if args.global_plots:
    # ============================================================
    # 3. Extract geo-parameters + pressure metrics
    # ============================================================
    records = []

    for vtp in vtp_files:
        run_id = vtp.parent.name.split("_")[1]
        csv_path = vtp.parent / f"geo_parameters_{run_id}.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        geo_df = pd.read_csv(csv_path)
        geo = geo_df.iloc[0].to_dict()

        mesh = pv.read(str(vtp))
        cp = mesh.cell_data.get("CpMeanTrim", mesh.point_data.get("CpMeanTrim", None))
        if cp is None:
            print(f"Warning: No CpMeanTrim in {vtp.name}")
            continue

        mean_cp = float(np.mean(cp))
        cp_range = float(np.max(cp) - np.min(cp))
        cp_std = float(np.std(cp))

        record = {
            "run": run_id,
            "mean_Cp": mean_cp,
            "Cp_range": cp_range,
            "Cp_std": cp_std,
            **geo
        }
        records.append(record)

    df = pd.DataFrame(records)
    print("\nExtracted data (first 5 rows):")
    print(df.head())

    # ============================================================
    # 4. Choose two most impactful geo-parameters
    # ============================================================
    possible_params = [" Vehicle_Ride_Height", " Rear_Diffusor_Angle",
                       " Vehicle_Length", " Vehicle_Width", " Vehicle_Height",
                       " Hood_Angle", " Windscreen_Angle", " Backlight_Angle"]

    param1 = next((p for p in possible_params if p in df.columns), None)
    param2 = next((p for p in possible_params if p in df.columns and p != param1), None)

    if param1 is None or param2 is None:
        print("Could not find enough geo-parameter columns!")
    else:
        print(f"\nUsing parameters: {param1}, {param2}")

        # ============================================================
        # 5. Two line plots (Dark Theme)
        # ============================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#1e1e1e")
        fig.patch.set_facecolor("#1e1e1e")

        df_sorted1 = df.sort_values(param1)
        axes[0].plot(df_sorted1[param1], df_sorted1["mean_Cp"], marker="o", linewidth=2, color="#00d4ff")
        axes[0].set_xlabel(param1, color="white")
        axes[0].set_ylabel("Mean Cp", color="white")
        axes[0].set_title(f"Mean Cp vs {param1}", color="white")
        axes[0].tick_params(colors="white")
        axes[0].set_facecolor("#1e1e1e")
        axes[0].spines['bottom'].set_color('white')
        axes[0].spines['left'].set_color('white')
        axes[0].spines['top'].set_color('white')
        axes[0].spines['right'].set_color('white')
        axes[0].grid(True, alpha=0.3, color="white")

        df_sorted2 = df.sort_values(param2)
        axes[1].plot(df_sorted2[param2], df_sorted2["Cp_range"], marker="s", color="#ff7f0e", linewidth=2)
        axes[1].set_xlabel(param2, color="white")
        axes[1].set_ylabel("Cp Range (max - min)", color="white")
        axes[1].set_title(f"Cp Range vs {param2}", color="white")
        axes[1].tick_params(colors="white")
        axes[1].set_facecolor("#1e1e1e")
        axes[1].spines['bottom'].set_color('white')
        axes[1].spines['left'].set_color('white')
        axes[1].spines['top'].set_color('white')
        axes[1].spines['right'].set_color('white')
        axes[1].grid(True, alpha=0.3, color="white")

        plt.tight_layout()
        plt.show()

print("\nDone.")