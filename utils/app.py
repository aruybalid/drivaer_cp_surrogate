"""
Streamlit Web App for DrivAer Cp Inference.
- Shows basic info about the uploaded file immediately
- Shows rotating training images while inference runs (decimation is slow)
- Accepts uploaded model (.pt)
- Tab 2 shows: Predicted Cp + Ground Truth Cp + Absolute Error
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pyvista as pv
from pathlib import Path
import tempfile
import time
from inference import predict_cp

st.set_page_config(page_title="DrivAer Cp Surrogate", layout="wide")
st.title("🚗 DrivAer Surface Cp Predictor")
st.caption("Upload a trained model (.pt) + geometry (.vtp) → Run inference")

# ============================================================
# Session state
# ============================================================
if 'raw_mesh' not in st.session_state:
    st.session_state.raw_mesh = None
if 'processed_mesh' not in st.session_state:
    st.session_state.processed_mesh = None
if 'infer_time' not in st.session_state:
    st.session_state.infer_time = None

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Inputs")

    uploaded_model = st.file_uploader("Upload trained model (.pt)", type=['pt'])
    uploaded_vtp = st.file_uploader("Upload geometry (.vtp)", type=['vtp'])
    use_demo = st.button("Use Held-out Demo (run_2)")

    target_faces = st.slider("Decimation target faces", 10000, 100000, 500000, step=10000)

    can_run = uploaded_model is not None and (uploaded_vtp is not None or use_demo)
    run_inference = st.button("▶ Run Inference", type="primary", disabled=not can_run)

# ============================================================
# Load raw mesh metadata only
# ============================================================
def load_raw_mesh_from_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    mesh = pv.read(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return mesh

if uploaded_vtp and st.session_state.raw_mesh is None:
    st.session_state.raw_mesh = load_raw_mesh_from_upload(uploaded_vtp)

if use_demo and st.session_state.raw_mesh is None:
    demo_path = Path("./references/run_2/boundary_2.vtp")
    if demo_path.exists():
        st.session_state.raw_mesh = pv.read(str(demo_path))
    else:
        st.error("Demo file not found. Please place a .vtp in references/run_2/.")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["Input Mesh", "Predicted Cp Field", "Diagnostics & Export"])

# ============================================================
# Tab 1: Basic info only
# ============================================================
with tab1:
    st.subheader("Input Mesh Information")
    raw = st.session_state.raw_mesh

    if raw is not None:
        st.write(f"**File loaded successfully**")
        st.write(f"- Points: {raw.n_points:,}")
        st.write(f"- Faces: {raw.n_faces:,}")
        st.info("Click 'Run Inference' to visualize the mesh and predicted Cp field.")
    else:
        st.info("Upload a .vtp file or click the demo button.")

# ============================================================
# Run Inference with Image Carousel (No Threading)
# ============================================================
if run_inference and uploaded_model:
    t0 = time.time()

    # Save uploaded model temporarily
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_model:
        tmp_model.write(uploaded_model.getvalue())
        model_path = tmp_model.name

    if use_demo:
        vtp_path = str(Path("./references/run_2/boundary_2.vtp"))
    else:
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as tmp_vtp:
            tmp_vtp.write(uploaded_vtp.getvalue())
            vtp_path = tmp_vtp.name

    # === Progress feedback with rotating images ===
    progress_placeholder = st.empty()
    status_text = st.empty()

    example_images = sorted(Path("./references/training_examples").glob("*.png"))

    if example_images:
        status_text.info("Running decimation + inference... This can take several minutes.")
        for i, img_path in enumerate(example_images * 2):   # loop multiple times
            progress_placeholder.image(str(img_path), width='content',
                                       caption=f"Training Example {i % len(example_images) + 1}")
            time.sleep(1.0)
    else:
        status_text.info("Running decimation + inference... Please wait.")
        for _ in range(10):
            time.sleep(2.0)

    # Actual inference
    processed_mesh, raw_data = predict_cp(
        vtp_path, model_path=model_path, target_faces=target_faces
    )

    # Cleanup
    Path(model_path).unlink(missing_ok=True)
    if not use_demo:
        Path(vtp_path).unlink(missing_ok=True)

    progress_placeholder.empty()
    status_text.empty()

    st.session_state.processed_mesh = processed_mesh
    st.session_state.infer_time = time.time() - t0
    st.success(f"Inference complete in {st.session_state.infer_time:.2f}s")

# ============================================================
# Tab 2: Predicted Cp + Ground Truth + Absolute Error
# ============================================================
with tab2:
    st.subheader("Predicted vs Ground Truth Cp Field")
    processed = st.session_state.processed_mesh

    if processed is not None and 'pred_Cp' in processed.cell_data:
        has_gt = 'GT_Cp' in processed.cell_data
        centers = processed.cell_centers().points

        if has_gt:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Predicted Cp**")
                fig1 = go.Figure(data=[go.Scatter3d(
                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=processed.cell_data['pred_Cp'],
                                colorscale='RdBu_r', cmin=-1.2, cmax=1.2,
                                colorbar=dict(title="Cp")),
                )])
                fig1.update_layout(scene=dict(aspectmode='data'), height=500)
                st.plotly_chart(fig1, width='content')

            with col2:
                st.markdown("**Ground Truth Cp**")
                fig2 = go.Figure(data=[go.Scatter3d(
                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                    mode='markers',
                    marker=dict(size=2, color=processed.cell_data['GT_Cp'],
                                colorscale='RdBu_r', cmin=-1.2, cmax=1.2,
                                colorbar=dict(title="Cp")),
                )])
                fig2.update_layout(scene=dict(aspectmode='data'), height=500)
                st.plotly_chart(fig2, width='content')

            # Absolute Error
            st.markdown("**Absolute Error**")
            error = processed.cell_data['error']
            fig_err = go.Figure(data=[go.Scatter3d(
                x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode='markers',
                marker=dict(size=2, color=error, colorscale='inferno',
                            cmin=0, cmax=2, colorbar=dict(title="|Error|")),
            )])
            fig_err.update_layout(scene=dict(aspectmode='data'), height=500)
            st.plotly_chart(fig_err, width='content')

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{error.mean():.4f}")
            with col2:
                st.metric("Max Absolute Error", f"{error.max():.4f}")
            with col3:
                st.metric("Median Absolute Error", f"{np.median(error):.4f}")

        else:
            st.markdown("**Predicted Cp**")
            fig = go.Figure(data=[go.Scatter3d(
                x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
                mode='markers',
                marker=dict(size=2, color=processed.cell_data['pred_Cp'],
                            colorscale='RdBu_r', cmin=-1.2, cmax=1.2,
                            colorbar=dict(title="Cp")),
            )])
            fig.update_layout(scene=dict(aspectmode='data'), height=500)
            st.plotly_chart(fig, width='content')

    else:
        st.info("Click 'Run Inference' to see the predicted Cp field.")

# ============================================================
# Tab 3: Diagnostics & Export
# ============================================================
with tab3:
    st.subheader("Diagnostics & Export")
    processed = st.session_state.processed_mesh

    if processed is not None:
        st.write(f"Inference time: {st.session_state.get('infer_time', 0):.2f} s")

        if 'pred_Cp' in processed.cell_data and 'GT_Cp' in processed.cell_data:
            cp = processed.cell_data['pred_Cp']
            gt = processed.cell_data['GT_Cp']
            error = processed.cell_data['error']

            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=cp, nbinsx=50, name='Predicted Cp', opacity=0.7))
            hist_fig.add_trace(go.Histogram(x=gt, nbinsx=50, name='Ground Truth Cp', opacity=0.7))
            hist_fig.update_layout(barmode='overlay', height=300, title="Cp Distribution")
            st.plotly_chart(hist_fig, width='content')

            error_hist = go.Figure(data=[go.Histogram(x=error, nbinsx=50, name='Absolute Error')])
            error_hist.update_layout(height=250, title="Absolute Error Distribution")
            st.plotly_chart(error_hist, width='content')

        if st.button("Export predicted .vtp"):
            out_path = "predicted_output.vtp"
            processed.save(out_path)
            with open(out_path, "rb") as f:
                st.download_button("Download predicted.vtp", f, file_name="predicted_Cp.vtp")
    else:
        st.info("Run inference first.")

st.markdown("---")
st.caption("Step 3 – Deploy | Upload model (.pt) and geometry (.vtp)")