# DrivAer Cp Physics-ML Surrogate
**Physics-Informed Surrogate Modeling Pipeline**  
An end-to-end Physics-AI framework that learns surface pressure distributions directly from 3D geometry, delivering near real-time aerodynamic predictions without running expensive CFD simulations.  

Built on the DrivAerML dataset (https://huggingface.co/datasets/neashton/drivaerml), this production-grade prototype covers the full workflow: mesh decimation → physics-aware feature engineering → neural surrogate training (PointNet/MLP) → interactive Streamlit inference app.

Given a surface geometry (.vtp file), a neural net, trained on CFD computations, predicts the surface **pressure coefficient** ($C_p$) in minutes:

$$
C_p = \frac{p - p_\infty}{\frac{1}{2} \rho U_\infty^2}
$$

Where:
- $p$ = local static pressure on the surface
- $p_\infty$ = freestream static pressure
- $\rho$ = freestream density
- $U_\infty$ = freestream velocity

**Core Idea**: 
-  Decimate surface meshes from boundary_i.vtp to ~5k points. 
-  Use a point cloud-based regressor of different forms (pure PyTorch, e.g., MLP/PointNet), taking normalized coords + normals to learn CpMeanTrim per point. 
-  Captures local geometry + global shape context (PointNet through global feature branch). 
-  Trains fast on laptop CPU/GPU. 
-  As a prototype: produces directionally correct fields (stagnation, acceleration zones) on held-out meshes. Trained only on 10 geometries (prototype).

**Why this approach**:
- Only surface geometry (points, normals derived from .vtp).
- Practical: decimation + lightweight model (MLP) for prototyping + limited resources.
- With limited training samples (only 10), **simplicity trumps complexity**: MLP is less prone to overfitting (tested against PointNet + global param branch), and more practical than full GNN for quick execution.

---

## Installation

### Using venv + pip
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using conda
```bash
conda create -n drivaer python=3.10 -y
conda activate drivaer
pip install -r requirements.txt
```

---

## Project Structure (Stepwise Workflow)

This project is organized into three sequential stages. Each stage has its own folder containing a `HOWTO.md` file that describes what the stage does and how to run it.

| Stage | Folder                        | Purpose                                      | Key Script(s)                     |
|-------|-------------------------------|----------------------------------------------|-----------------------------------|
| 1     | `1_Data_Processing/`          | Load, inspect, decimate and store data       | `plot_data.py`, `preprocess.py`, `plot_processed.py` |
| 2     | `2_Surrogate_Building/`       | Train surrogate models and run inference     | `train.py`, `inference.py`        |
| 3     | `3_Deploy/`                   | Launch the interactive Streamlit web app     | `app.py`                          |

**Important**: Complete the stages in order. Human review is expected between stages (especially after Stage 1).

---

## Stage 1: Data Processing

**Location**: `1_Data_Processing/`

**What it does**:
- Plot the raw surface data from the `./references` folder.
- Preprocess (decimate + select variables such as `CpMeanTrim`).
- Store the processed data in `./output/processed`.
- Plot the processed data for human review.

**How to run**:
```bash
cd 1_Data_Processing
python ../utils/plot_data.py
python ../utils/preprocess.py --target_faces 5000
python ../utils/plot_processed.py
```

Human review of the processed data in `./output/processed` is required before moving to Stage 2.

---

## Stage 2: Surrogate Building

**Location**: `2_Surrogate_Building/`

This stage contains two sub-steps:

### 2.1 Surrogate Training (`21_Surrogate_Training/`)
- Trains different model architectures (PointNet, MLP variants, with or without geometric parameters) on the preprocessed data.
- Models are defined in `../utils/model.py`.

**How to run**:
```bash
cd 2_Surrogate_Building/21_Surrogate_Training
python ../../utils/train.py --model pointnet
python ../../utils/train.py --model mlp_no_global --epochs 100
python ../../utils/train.py --model mlp_with_params --hidden 256
python ../../utils/train.py --model pointnet_with_params --batch_size 4
```

### 2.2 Surrogate Testing (`22_Surrogate_Testing/`)
- Runs inference on held-out geometries using a trained model.
- Generates predicted `.vtp` files and visualizes results.

**How to run**:
```bash
cd 2_Surrogate_Building/22_Surrogate_Testing
python ../../utils/inference.py --vtp_path ./references/data/run_1/boundary_1.vtp \
    --model_path ./references/best_model.pt \
    --model pointnet_with_params \
    --target_faces 10000000000
python ../../utils/plot_prediction.py --vtp ./output/predicted_....vtp
```

---

## Stage 3: Deployment

**Location**: `3_Deploy/`

Launches the interactive Streamlit web application that allows users to upload a trained model and run inference on new geometries.

**How to run**:
```bash
cd 3_Deploy
streamlit run ../utils/app.py --server.maxUploadSize=1000 --server.maxMessageSize=1000
```

The app supports multiple model architectures (selected via dropdown) and includes visualization of predicted Cp, ground truth, absolute error, and relative error.

---

## Data Requirements

Download only the required subset (~5–6 GB) from Hugging Face:

**Training geometries**: `run_80, 98, 102, 208, 213, 281, 397, 425, 445, 474`
**Held-out test**: e.g. `run_1` or `run_50`

## Key Engineering Highlights

- **Lightweight surrogate**: PointNet / MLP trained on only 10 geometries → directional accuracy on unseen shapes.
- **Production-ready inference**: Streamlit app with model switching, error visualization, and export capabilities.
- **Practical trade-offs**: Mesh decimation to ~5k points + simple architectures chosen for speed and robustness on limited data.
- **Physics-aware features**: Surface normals + global shape context used as input; CpMeanTrim learned as target.

## Results

The interactive Streamlit app (`3_Deploy/`) provides visual evaluation of model predictions, including side-by-side comparison of predicted vs. ground-truth $C_p$ fields, absolute error, and relative error maps. This allows direct assessment of surrogate quality on held-out geometries.