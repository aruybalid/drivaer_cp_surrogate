# Demo Outline (3-5 slides / bullets for GTM presentation)

**Slide 1: The Opportunity**
- CFD is slow/expensive for early design iterations.
- This prototype: geometry-only surface Cp prediction in <5s on laptop from .vtp mesh.
- Value: rapid what-if studies, democratize aero insights for design teams.

**Slide 2: Technical Approach (high-level for customers)**
- Trained exclusively on 10 DrivAerML surface cases (boundary_i.vtp).
- Preprocess: PyVista decimate to 5k points + compute normals.
- Model: Lightweight PointNet (pure PyTorch) — local geometry features + global shape context via max-pooling.
- Learned non-trivial mapping: produces directionally correct Cp (front high pressure, roof low, wake effects).

**Slide 3: Live Walkthrough**
- Upload or demo held-out .vtp.
- One-click inference.
- Interactive 3D viewer: predicted Cp field (color-mapped, rotatable).
- Side panel: stats, histogram match to expected physics, error if GT available.
- Export predicted mesh for further analysis.

**Slide 4: Results & Limitations**
- On held-out: not noise/constant — captures macro patterns from geometry variation.
- Tradeoffs accepted: aggressive decimation for speed; small training set (prototype).
- Path forward: full dataset + PhysicsNeMo GLOBE / MeshGraphNets for higher fidelity + uncertainty.

**Slide 5: Why This Wins for Customers**
- Intuitive UI focused on actionable visuals + diagnostics.
- No CFD expertise or licenses needed for inference.
- Extensible: add force integration, multiple fields, batch mode.
- Demonstrates solid engineering judgment: practical scope, clear physics-ML ideas, customer empathy in UX.

(Keep demo under 7 min; invite questions on model choices during technical round.)