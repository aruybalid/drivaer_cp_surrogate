# DrivAer Cp Surrogate - Physics-ML Take-Home

Practical end-to-end solution for Surface-Only Aerodynamics ML + Inference Web App.

**Core Idea**: Decimate surface meshes from boundary_i.vtp to ~5k points. Use PointNet-style point cloud regressor (pure PyTorch) taking normalized coords + normals to predict CpMeanTrim per point. Captures local geometry + global shape context. Trains fast on laptop CPU/GPU. Produces directionally correct fields (stagnation, acceleration zones) on held-out meshes.

**Why this approach**:
- Only surface geometry (points, normals derived from vtp).
- Practical: decimation + lightweight model fits 1-week scope + limited resources.
- Physics-ML flavor: respects mesh geometry via point cloud + normals; equivariant-ish via normalization.
- Better than pure MLP (global context via PointNet pooling); simpler than full GNN/PhysicsNeMo GLOBE for quick execution.
- Absolute accuracy secondary; focus on learning non-trivial mapping.

## Setup (one-time)
```bash
cd drivaer_cp_surrogate
python -m venv venv
source venv/bin/activate  # or conda
pip install -r requirements.txt
```

## Data Prep (download only needed ~5-6 GB)
Use HuggingFace to get exactly the 10 train runs + 1-2 held-out test (e.g. run_1 or run_50).

See `scripts/download_data.sh` (create it) or run in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="neashton/drivaerml",
    repo_type="dataset",
    local_dir="./data",
    allow_patterns=["run_80/boundary_*.vtp", "run_98/boundary_*.vtp", ... ]  # add the 10 + test
)
```
List of train: run_80,98,102,208,213,281,397,425,445,474. Pick e.g. run_1 as held-out test.

## Preprocess & Train
```bash
python preprocess.py --data_dir ./data --out_dir ./processed --target_faces 8000
python train.py --processed_dir ./processed --epochs 200 --batch_size 2 --lr 1e-3
```
- Saves model to models/best_model.pt
- Logs train/val loss (use 8/2 split or all for demo).
- With augmentation (jitter, dropout) to combat small data.

## Run Web App (Streamlit)
```bash
streamlit run app.py
```
- Upload any .vtp (or use built-in held-out example button).
- Preprocess on-the-fly (same pipeline).
- Inference: predict Cp field.
- UI: 3D interactive point cloud (Plotly) colored by pred Cp vs GT if available; histograms, stats (mean Cp, range), diagnostics (mesh size, inference time).
- Export predicted .vtp for Paraview inspection.
- Thoughtful UX: clear tabs (Input | Prediction | Diagnostics), color scales matching CFD convention (blue low, red high pressure), quick "Run Inference" button, explanations.

## Demo Outline (for GTM / customer walkthrough, 5-7 min)
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

## Files
- `preprocess.py`: PyVista load/decimate, extract points/normals/Cp, normalize, save .pt dicts.
- `train.py`: PointNet model, Dataset, train loop with MSE loss + simple aug.
- `inference.py`: Reusable preprocess + model forward for new vtp.
- `app.py`: Streamlit UI with tabs, Plotly viz, export.
- `model.py`: PointNet definition (clean, commented).
- `requirements.txt`
- `demo_slides.md` (expand bullets to 3-5 slides if needed).

## Held-out Test
After training, `python inference.py --vtp_path data/run_XXX/boundary_XXX.vtp --model models/best_model.pt --out predicted.vtp`

Expect: directional correctness (not noise/constant). Visualize in Paraview: CpMeanTrim vs pred_Cp.

## Tradeoffs & Justifications (for technical interview)
- Decimation: necessary for laptop training; loses fine details but captures macro pressure patterns.
- PointNet vs GNN: simpler deps/install, sufficient for prototype; GNN would use mesh edges explicitly but more complex.
- No PhysicsNeMo/GLOBE: faster to implement/train on CPU; GLOBE excellent for production (equivariant, discretization invariant) but heavier setup.
- Small data: augmentation + early stopping key; shows idea works.
- Web app: focused on customer experience (intuitive, visual, actionable diagnostics) over fancy hosting.

Ready for demo. Good luck! (Tailored to Rescale-style engineering AI role.)