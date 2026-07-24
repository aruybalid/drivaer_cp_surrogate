# DrivAer Cp Physics-ML Surrogate

Practical end-to-end solution for Surface-Only Aerodynamics ML + Inference Web App.

**Core Idea**: Decimate surface meshes from boundary_i.vtp to ~5k points. Use a point cloud-based regressor of different forms (pure PyTorch, e.g., MLP, PointNet), taking normalized coords + normals to predict CpMeanTrim per point. Captures local geometry + global shape context. Trains fast on laptop CPU/GPU. Produces directionally correct fields (stagnation, acceleration zones) on held-out meshes.

**Why this approach**:
- Only surface geometry (points, normals derived from vtp).
- Practical: decimation + lightweight model fits 1-week scope + limited resources.
- Physics-ML flavor: respects mesh geometry via point cloud + normals; equivariant-ish via normalization.
- Better than pure MLP (global context via PointNet pooling); simpler than full GNN/PhysicsNeMo for quick execution.
- Absolute accuracy secondary; focus on learning non-trivial mapping.

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

## Setup Instructions

### Setup (one-time)
```bash
cd drivaer_cp_surrogate
python -m venv venv
source venv/bin/activate  # or conda
pip install -r requirements.txt
```

### Data Prep (download only needed ~5-6 GB)
Use HuggingFace to download exactly the 10 train runs + 1-2 held-out test (e.g. run_1 or run_50).

List of train: run_80,98,102,208,213,281,397,425,445,474. Pick e.g. run_1 as held-out test.

## Demo Outline (for customer walkthrough, 5-7 min)
1. **Problem & Value**: "Traditional CFD takes hours/days per design iteration. This surrogate predicts surface Cp in seconds from geometry only — enabling rapid exploration in early design."
2. **Approach (keep high-level)**: "Trained PointNet regressor on decimated surface meshes from 10 DrivAerML cases. Inputs: point coords + normals (pure geometry). Learned to map shape variations to realistic pressure distributions."
3. **Live Demo**:
   - Load held-out .vtp (or upload).
   - Click "Run Inference".
   - Show side-by-side: input mesh | predicted Cp field (interactive 3D, rotate/zoom).
   - Highlight: stagnation at front (~Cp=1), low pressure on roof/fast areas, base wake effects — directionally matches physics/CFD expectations.
   - Diagnostics panel: error vs GT (if avail), Cp histogram match, integrated rough force proxy.
4. **Limitations & Next**: "With 10 samples, it's a prototype — shows generalization across parametric variants. Scale to full 400+ with GLOBE/PhysicsNeMo or MeshGraphNets for production accuracy. Add uncertainty, full Cf prediction, volume fields later."
5. **Why compelling for customer**: Fast what-if on new geometries, integrates into design loop, no CFD license needed for inference. Tech stack: PyTorch (train) + Streamlit (deployable app).

### Files
- `preprocess.py`: PyVista load/decimate, extract points/normals/Cp, normalize, save .pt dicts.
- `train.py`: PointNet model, Dataset, train loop with MSE loss + simple aug.
- `inference.py`: Reusable preprocess + model forward for new vtp.
- `app.py`: Streamlit UI with tabs, Plotly viz, export.
- `model.py`: PointNet definition (clean, commented).
- `requirements.txt`
- `demo_slides.md` (expand bullets to 3-5 slides if needed).

### Held-out Test
After training, `python inference.py --vtp_path data/run_XXX/boundary_XXX.vtp --model models/best_model.pt --out predicted.vtp`

Expect: directional correctness (not noise/constant). Visualize in Paraview: CpMeanTrim vs pred_Cp.

### Tradeoffs & Justifications
- Decimation: necessary for laptop training; loses fine details but captures macro pressure patterns.
- PointNet vs GNN: simpler deps/install, sufficient for prototype; GNN would use mesh edges explicitly but more complex.
- No PhysicsNeMo/GLOBE: faster to implement/train on CPU; GLOBE excellent for production (equivariant, discretization invariant) but heavier setup.
- Small data: augmentation + early stopping key; shows idea works.
- Web app: focused on customer experience (intuitive, visual, actionable diagnostics) over fancy hosting.

## Customer Walkthrough
### Slide 1: The Challenge with Traditional CFD
**Title: Traditional CFD is Too Slow for Modern Design Cycles**

Aerodynamic design today relies heavily on Computational Fluid Dynamics (CFD)

A single high-fidelity CFD simulation can take hours to days to complete

This creates a major bottleneck in the design process:
- Limited number of design iterations
- High computational cost
- Delayed decision-making

As a result, engineers often have to make critical design choices with incomplete information

Key Message:
We need a much faster way to understand surface pressure behavior during early-stage design.

### Slide 2: Our Approach – Fast Surrogate Modeling
**Title: A Physics-Informed Surrogate That Predicts Surface Pressure in Seconds** 

We built a machine learning model that predicts surface pressure coefficient (Cp) directly from geometry
The model uses a PointNet-style neural network trained on high-fidelity CFD data

Key technical choices:
Works on surface geometry only (no volume mesh required)
Uses decimated meshes (~15k points) for fast inference
Combines local geometry features with global shape context
The result: accurate, directional predictions in just a few seconds

Key Message:
A practical balance between speed and physical fidelity.
---

### Slide 3: Results & Business Impact
**Title: Fast, Actionable Insights That Accelerate Design**

Speed: Inference in seconds vs. hours/days with traditional CFD
Usability: Interactive web app allows engineers to upload geometries and instantly see predicted pressure fields
Performance: Strong directional accuracy on unseen geometries (stagnation, acceleration, wake effects)
Scalability: Model trained on only 10 cases yet generalizes well across parametric variants
Business Value:

Enable 10x–100x more design iterations in early phases
Reduce reliance on expensive CFD licenses for early exploration
Support faster, more confident design decisions
Next Steps: Scale to larger datasets and explore uncertainty quantification.

Suggested Slide Titles (for your deck).  
- The Bottleneck in Aerodynamic Design  
- A Fast, Geometry-Driven Surrogate Model  
- Accelerating Design Through Instant Predictions  