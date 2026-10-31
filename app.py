"""
PlantHealth AI - AgTech Disease Classification Web Dashboard
High-performance, professional Streamlit application for automated
plant disease diagnosis using Deep Convolutional Neural Networks.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure page layout and AgTech styling
st.set_page_config(
    page_title="PlantHealth AI | Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom AgTech CSS
CUSTOM_CSS = """
<style>
    /* Main Background & Fonts */
    .main {
        background-color: #F8FAF8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Styling */
    .hero-banner {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(27, 94, 32, 0.15);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.95;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }
    
    /* AgTech Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid #E0E7E1;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(46, 125, 50, 0.08);
    }
    .metric-val {
        font-size: 2.0rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0.2rem;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #556B58;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Diagnosis Pill Badges */
    .badge-healthy {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .badge-diseased {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FFCDD2;
    }

    /* Section Cards */
    .card-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Imports from src
from config import (
    CLASS_INDICES_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    CUSTOM_CNN_MODEL_PATH,
    DATA_DIR,
    DEMO_SAMPLES_DIR,
    EVAL_REPORT_PATH,
    FINAL_CLASSIFICATION_REPORT_PATH,
    FINAL_CONFUSION_MATRIX_PATH,
    FINAL_METRICS_JSON_PATH,
    FINAL_MODEL_COMPARISON_PATH,
    FINAL_PREDICTION_SAMPLES_PATH,
    FINAL_RESULTS_DIR,
    FINAL_TRAINING_HISTORY_PATH,
    METRICS_JSON_PATH,
    MOBILENET_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    PREDICTION_SAMPLES_PATH,
    RESULTS_DIR,
    TRAINING_HISTORY_PATH,
    VGG16_MODEL_PATH,
)
from src.predict import PlantDiseasePredictor
from src.preprocessing import ImageValidationError, validate_and_load_image
from src.utils import load_json, parse_class_name


@st.cache_resource
def get_cached_predictor(model_choice: str) -> PlantDiseasePredictor:
    """Caches loaded model in memory across Streamlit reruns."""
    if model_choice == "MobileNetV2 (Primary)":
        return PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH, model_type="mobilenetv2")
    elif model_choice == "Custom CNN":
        return PlantDiseasePredictor(model_path=CUSTOM_CNN_MODEL_PATH, model_type="custom_cnn")
    elif model_choice == "VGG16 Benchmark":
        return PlantDiseasePredictor(model_path=VGG16_MODEL_PATH, model_type="vgg16")
    return PlantDiseasePredictor(model_path=MOBILENET_MODEL_PATH, model_type="mobilenetv2")


def render_sidebar() -> Tuple[str, str]:
    """Renders persistent sidebar navigation and model selector."""
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?w=500&auto=format&fit=crop&q=60",
                 use_container_width=True)
        st.markdown("## 🌿 PlantHealth AI")
        st.caption("Deep Learning Plant Pathology System")

        nav = st.radio(
            "Navigation",
            ["Home Dashboard", "Disease Detection", "Model Performance", "About & Architecture"],
            index=1  # Default to Disease Detection for quick demo
        )

        st.markdown("---")
        st.markdown("### ⚙️ Inference Engine")
        available_models = ["MobileNetV2 (Primary)", "Custom CNN"]
        if VGG16_MODEL_PATH.exists():
            available_models.append("VGG16 Benchmark")

        selected_model = st.selectbox("Active Model Architecture", available_models, index=0)

        # Check status indicator
        model_path = MOBILENET_MODEL_PATH if "MobileNet" in selected_model else CUSTOM_CNN_MODEL_PATH
        if model_path.exists():
            st.success(f"● Model Active: {selected_model}")
        else:
            st.warning(f"○ Checkpoint Missing: Train model first")

        st.markdown("---")
        st.caption("PlantVillage 38-Class Dataset | TensorFlow 2.x")

    return nav, selected_model


def render_home_dashboard():
    """Renders project overview and operational metrics."""
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🌿 PlantHealth AI Dashboard</div>
        <div class="hero-subtitle">
            An enterprise-grade computer vision system engineered for automated crop disease diagnosis,
            foliar pathogen classification, and instant preventive agricultural guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">38</div>
            <div class="metric-lbl">Diagnostic Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">14</div>
            <div class="metric-lbl">Crops & Species</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # Load accuracy from metrics.json if available
        acc_text = "N/A"
        target_path = FINAL_METRICS_JSON_PATH if FINAL_METRICS_JSON_PATH.exists() else METRICS_JSON_PATH
        if target_path.exists():
            try:
                metrics = load_json(target_path)
                primary = metrics.get("MobileNetV2 (Primary)", {})
                if "accuracy" in primary:
                    acc_text = f"{primary['accuracy']:.1%}"
            except Exception:
                pass
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{acc_text}</div>
            <div class="metric-lbl">Test Accuracy (Primary)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">224px</div>
            <div class="metric-lbl">Native Resolution</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow Cards
    st.subheader("🌾 How the Automated Diagnosis Pipeline Operates")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">1. Foliar Image Intake</div>
            <p>Upload a high-resolution leaf photograph. The pipeline performs automated format validation,
            corrupted file rejection, EXIF orientation correction, and standardizes dimensions to 224 × 224 RGB.</p>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">2. CNN Feature Extraction</div>
            <p>The input passes through MobileNetV2 inverted residual blocks with linear bottlenecks, extracting
            fine spatial patterns of fungal mycelium, bacterial lesions, chlorosis, and necrotic halos.</p>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="card-box">
            <div class="card-header">3. Agronomic Advisory</div>
            <p>Yields calibrated probability distributions over all 38 classes, mapping the prediction to plant taxonomy,
            pathogen etiology, and actionable agronomic prevention recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

    # Agricultural Field Deployment Disclaimer
    st.warning(
        "⚠️ **Agricultural Field Deployment Disclaimer:** Benchmark evaluation metrics are established on the standardized "
        "laboratory-controlled PlantVillage dataset (uniform backgrounds, single centered leaf, controlled lighting). "
        "High benchmark performance under laboratory conditions does not guarantee equivalent diagnostic accuracy in "
        "real agricultural field environments, where complex soil backgrounds, fluctuating sunlight, leaf occlusions, "
        "and co-occurring nutritional deficiencies are present. Field deployments require on-site validation."
    )


def render_disease_detection(selected_model: str):
    """Renders interactive disease classification workstation."""
    st.markdown("## 🔬 Plant Disease Diagnosis Workstation")
    st.write(
        "Upload a leaf image or select from sample agricultural specimens to diagnose plant health and receive treatment advice."
    )

    st.info(
        "💡 **Field Notice:** For highest diagnostic accuracy, photograph a single leaf centered against a plain or shaded "
        "background to approximate training conditions."
    )

    predictor = get_cached_predictor(selected_model)

    if not predictor.is_ready:
        st.error(
            f"⚠️ Model checkpoint `{selected_model}` is not yet available on disk.\n\n"
            f"To train the model, run:\n"
            f"```bash\npython -m src.train --model mobilenetv2\n```"
        )
        return

    # Initialize session state for persistent interactive diagnosis
    if "active_specimen" not in st.session_state:
        st.session_state["active_specimen"] = None
        st.session_state["active_label"] = None

    # Retrieve pre-loaded demo specimens (available on Streamlit Cloud without local dataset)
    demo_files = sorted(list(DEMO_SAMPLES_DIR.glob("*.jpg")) + list(DEMO_SAMPLES_DIR.glob("*.png")))

    # Tabs for Upload vs Sample Gallery
    tab1, tab2 = st.tabs(["📁 Upload Leaf Image", "🖼️ Interactive Specimen Gallery (16 Pre-loaded)"])

    with tab1:
        col_up1, col_up2 = st.columns([1.2, 1])
        with col_up1:
            uploaded_file = st.file_uploader(
                "Upload a leaf photograph (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                help="Accepts smartphone photographs or field leaf scans"
            )
            if uploaded_file is not None:
                st.session_state["active_specimen"] = uploaded_file
                st.session_state["active_label"] = None

        with col_up2:
            st.markdown("##### ⚡ Quick Preset Test")
            st.caption("No image on your device? Select an instant pre-loaded specimen:")
            if demo_files:
                demo_options = {
                    f"{parse_class_name(p.stem)['plant']} — {parse_class_name(p.stem)['disease']}": p
                    for p in demo_files
                }
                preset_name = st.selectbox(
                    "Choose an agricultural specimen:",
                    ["-- Select a demo leaf --"] + list(demo_options.keys()),
                    index=0
                )
                if preset_name != "-- Select a demo leaf --":
                    st.session_state["active_specimen"] = str(demo_options[preset_name])
                    st.session_state["active_label"] = demo_options[preset_name].stem

    with tab2:
        st.markdown("##### 🌿 Click any leaf specimen below to diagnose instantly:")
        st.caption("16 representative agricultural crops and diseases pre-packaged for instant cloud demonstration:")

        sample_paths = demo_files
        if not sample_paths and Path(DATA_DIR).exists():
            sample_paths = list(Path(DATA_DIR).glob("*/*.JPG"))

        if sample_paths:
            # Display in responsive grid of 4 columns
            cols = st.columns(4)
            for i, p in enumerate(sample_paths):
                col = cols[i % 4]
                with col:
                    cls_name = p.stem if p.parent == DEMO_SAMPLES_DIR else p.parent.name
                    parsed = parse_class_name(cls_name)
                    is_healthy = parsed["status"] == "Healthy"
                    badge_style = "color: #1B5E20; background: #E8F5E9;" if is_healthy else "color: #B71C1C; background: #FFEBEE;"

                    try:
                        img_thumb = Image.open(p).resize((160, 160))
                        st.image(img_thumb, use_container_width=True)
                    except Exception:
                        pass

                    st.markdown(f"""
                    <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 2px;">{parsed['plant']}</div>
                    <div style="font-size: 0.78rem; font-weight: 500; {badge_style} padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">
                        {parsed['disease'][:22]}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🔬 Diagnose", key=f"demo_btn_{i}", use_container_width=True):
                        st.session_state["active_specimen"] = str(p)
                        st.session_state["active_label"] = cls_name
                        st.rerun()
        else:
            st.info("No demo specimens found. Please add sample images to `demo_samples/`.")

    selected_image = st.session_state.get("active_specimen")
    image_label_hint = st.session_state.get("active_label")

    # Image Preview & Analysis Execution
    if selected_image is not None:
        st.markdown("---")
        c_hdr1, c_hdr2 = st.columns([3, 1])
        with c_hdr1:
            st.markdown("### 📷 Active Leaf Diagnosis")
        with c_hdr2:
            if st.button("🔄 Reset / Clear Specimen", use_container_width=True):
                st.session_state["active_specimen"] = None
                st.session_state["active_label"] = None
                st.rerun()

        col_img, col_diag = st.columns([1, 1.3])

        with col_img:
            st.markdown("#### 🌿 Foliar Specimen")
            try:
                img_array, pil_img = validate_and_load_image(selected_image)
                st.image(pil_img, caption="Input Specimen (224 × 224 Lanczos Resized)", use_container_width=True)
                if image_label_hint:
                    parsed_hint = parse_class_name(image_label_hint)
                    st.info(f"**Expected Benchmark Label:** `{parsed_hint['plant']} — {parsed_hint['disease']}`")
            except ImageValidationError as e:
                st.error(f"Image Validation Error: {e}")
                return

        with col_diag:
            st.markdown("#### 🩺 Diagnostic Report")
            with st.spinner("Analyzing foliar pathology with deep neural network..."):
                try:
                    result = predictor.predict(selected_image, top_k=3)
                except Exception as e:
                    st.error(f"Inference Failure: {e}")
                    return

            best = result["top_prediction"]
            is_healthy = best["status"] == "Healthy"

            # Status Banner
            if is_healthy:
                st.markdown(f"""
                <div style="background-color: #E8F5E9; border-left: 6px solid #2E7D32; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <span style="font-size: 1.3rem; font-weight: 800; color: #1B5E20;">✅ HEALTHY FOLIAGE DETECTED</span><br>
                    <span style="color: #2E7D32; font-weight: 500;">No significant pathogenic foliar lesions or fungal colonies observed.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #FFEBEE; border-left: 6px solid #C62828; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <span style="font-size: 1.3rem; font-weight: 800; color: #B71C1C;">⚠️ PATHOLOGY CONFIRMED: {best['disease'].upper()}</span><br>
                    <span style="color: #C62828; font-weight: 500;">Infection detected on {best['plant']} foliage. Action required.</span>
                </div>
                """, unsafe_allow_html=True)

            # Core Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Crop Species", best["plant"])
            with m2:
                st.metric("Condition", best["disease"])
            with m3:
                st.metric("Confidence", best["percentage"])

            st.progress(float(best["confidence"]))

            # Pathogen and Clinical Description
            with st.expander("📖 Pathogen & Clinical Description", expanded=True):
                st.write(f"**Associated Pathogen:** *{best['pathogen']}*")
                st.write(best["description"])

            # Agricultural Prevention Recommendations
            with st.expander("🛡️ Actionable Agricultural Prevention & Treatment", expanded=True):
                for tip in best["prevention"]:
                    st.markdown(f"- {tip}")

            # Top 3 Candidates
            st.markdown("##### 🔍 Top-3 Diagnostic Candidates")
            top_df = pd.DataFrame([
                {
                    "Rank": f"#{i+1}",
                    "Crop": p["plant"],
                    "Condition": p["disease"],
                    "Status": p["status"],
                    "Confidence": p["percentage"]
                }
                for i, p in enumerate(result["top_k_predictions"])
            ])
            st.dataframe(top_df, hide_index=True, use_container_width=True)


def render_model_performance():
    """Renders evaluation graphs, confusion matrix, and benchmark comparison."""
    st.markdown("## 📊 Model Evaluation & Benchmarks")
    st.write("Rigorous quantitative metrics computed exclusively against the untouched 15% stratified test split.")

    # Metric Cards
    target_metrics_path = FINAL_METRICS_JSON_PATH if FINAL_METRICS_JSON_PATH.exists() else METRICS_JSON_PATH
    if target_metrics_path.exists():
        try:
            metrics_data = load_json(target_metrics_path)
            models = list(metrics_data.keys())
            st.markdown("### 🏆 Architecture Comparison")

            rows = []
            for m_name, m_vals in metrics_data.items():
                rows.append({
                    "Architecture": m_name,
                    "Test Accuracy": f"{m_vals.get('accuracy', 0):.2%}",
                    "Macro Precision": f"{m_vals.get('precision', 0):.2%}",
                    "Macro Recall": f"{m_vals.get('recall', 0):.2%}",
                    "Macro F1-Score": f"{m_vals.get('f1_score', 0):.2%}",
                    "Weighted F1": f"{m_vals.get('f1_weighted', 0):.2%}",
                    "Test Samples": f"{m_vals.get('total_test_samples', 0):,}"
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not parse metrics.json: {e}")
    else:
        st.info("Run `python -m src.evaluate` to generate test metrics and comparison charts.")

    st.markdown("---")

    # Visualizations Tabs
    vt1, vt2, vt3, vt4, vt5, vt6 = st.tabs([
        "📈 Training History",
        "🧩 Confusion Matrix",
        "⚖️ Confidence Calibration",
        "🔬 Qualitative Case Studies",
        "📊 Model Comparison",
        "📋 Per-Class Report"
    ])

    with vt1:
        st.markdown("#### Convergence & Loss Curves")
        p_hist = FINAL_TRAINING_HISTORY_PATH if FINAL_TRAINING_HISTORY_PATH.exists() else TRAINING_HISTORY_PATH
        if p_hist.exists():
            st.image(str(p_hist), use_container_width=True)
        else:
            st.info("Training history plot will appear here once model training finishes.")

    with vt2:
        st.markdown("#### 38-Class Full Test Confusion Matrix (13,208 Images)")
        p_cm_full = FINAL_RESULTS_DIR / "full_test_confusion_matrix.png"
        p_cm = p_cm_full if p_cm_full.exists() else (FINAL_CONFUSION_MATRIX_PATH if FINAL_CONFUSION_MATRIX_PATH.exists() else CONFUSION_MATRIX_PATH)
        if p_cm.exists():
            st.image(str(p_cm), use_container_width=True)
        else:
            st.info("Confusion matrix heatmap will appear here after evaluation.")

    with vt3:
        st.markdown("#### Probabilistic Confidence Calibration (Temperature Scaling)")
        st.write("Evaluates whether predicted softmax probabilities correspond to true empirical accuracy.")
        p_calib = FINAL_RESULTS_DIR / "calibration_reliability_diagram.png"
        if p_calib.exists():
            st.image(str(p_calib), use_container_width=True)
            st.caption("Left: Uncalibrated Softmax (ECE = 3.23%). Right: Post-Temperature Scaling T*=1.25 (ECE = 1.78%).")
        else:
            st.info("Calibration diagram will appear here after running `scripts/audit_confidence_calibration.py`.")

    with vt4:
        st.markdown("#### Unseen Test Prediction Grid (Correct vs. Misclassified)")
        p_samples = FINAL_RESULTS_DIR / "inference_validation_samples.png"
        p_qual = p_samples if p_samples.exists() else (FINAL_PREDICTION_SAMPLES_PATH if FINAL_PREDICTION_SAMPLES_PATH.exists() else PREDICTION_SAMPLES_PATH)
        if p_qual.exists():
            st.image(str(p_qual), use_container_width=True)
        else:
            st.info("Prediction sample grid will appear here after evaluation.")

    with vt5:
        st.markdown("#### Architecture Comparison (MobileNetV2 vs. Custom CNN)")
        p_comp = FINAL_MODEL_COMPARISON_PATH if FINAL_MODEL_COMPARISON_PATH.exists() else MODEL_COMPARISON_PATH
        if p_comp.exists():
            st.image(str(p_comp), use_container_width=True)
        else:
            st.info("Model comparison plot will appear here.")

    with vt6:
        st.markdown("#### Per-Class Diagnostic Performance Breakdown")
        p_rep_full = FINAL_RESULTS_DIR / "full_test_classification_report.csv"
        p_rep = p_rep_full if p_rep_full.exists() else (FINAL_CLASSIFICATION_REPORT_PATH if FINAL_CLASSIFICATION_REPORT_PATH.exists() else CLASSIFICATION_REPORT_PATH)
        if p_rep.exists():
            df_rep = pd.read_csv(p_rep, index_col=0)
            st.dataframe(df_rep.style.format("{:.2%}", subset=["precision", "recall", "f1-score"]),
                         use_container_width=True)
        else:
            st.info("Classification report CSV not found. Run `python -m src.evaluate`.")


def render_about_project():
    """Renders comprehensive architectural and methodology documentation."""
    st.markdown("## 📖 About PlantHealth AI")
    st.markdown(r"""
    ### 1. Problem Statement
    Plant diseases jeopardize over **20-40% of global agricultural food production** annually.
    Traditional laboratory diagnosis requires expert agronomists, culturing pathogens, or molecular PCR assays,
    which are inaccessible to rural smallholder farmers. PlantHealth AI provides real-time, non-invasive,
    edge-deployable disease identification directly from leaf imagery.

    ---

    ### 2. Model Architectures & Design
    - **Primary Model (MobileNetV2):** Leverages inverted residual blocks with linear bottlenecks.
      Pretrained on ImageNet ($1.4M$ images) to capture universal visual primitives (edges, gradients, textures).
      Custom top layers with Global Average Pooling, Batch Normalization, and Dropout ($0.4$) provide robust classification.
    - **Custom CNN (From Scratch):** 4-stage convolutional hierarchy ($32 \\to 64 \\to 128 \\to 256$ filters) with
      ReLU activations, Batch Normalization, Max-Pooling, and spatial Dropout to benchmark inductive bias without transfer priors.
    - **VGG16 Benchmark:** Classic deep 16-layer feed-forward benchmark.

    ---

    ### 3. Leakage-Free Dataset Engineering
    - **Dataset:** PlantVillage benchmark containing **87,000+ images across 38 classes**.
    - **Anti-Leakage Strategy:** Images originating from the same physical leaf (including multiple orientations,
      rotations, and zooms) share root UUID hashes. Splits strictly segregate whole sample groups into
      **70% Train, 15% Validation, and 15% Test**.
    - **Runtime Augmentation:** Applied strictly during training (`training=True`) with random rotation ($\pm 15^\circ$),
      horizontal flips, translation, and brightness scaling. Test and validation images remain pristine.

    ---

    ### 4. Technical Stack
    - **Deep Learning:** TensorFlow 2.x, Keras 3.x
    - **Computer Vision:** OpenCV, Pillow, Scikit-learn
    - **Web Framework:** Streamlit
    - **Runtime:** Python 3.11 on Apple Silicon / CPU
    """)


def main():
    nav, selected_model = render_sidebar()

    if nav == "Home Dashboard":
        render_home_dashboard()
    elif nav == "Disease Detection":
        render_disease_detection(selected_model)
    elif nav == "Model Performance":
        render_model_performance()
    elif nav == "About & Architecture":
        render_about_project()


if __name__ == "__main__":
    main()
