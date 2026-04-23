"""
app.py — Pneumonia Detection Streamlit Application
Run: streamlit run app.py
"""

import os, json, sys
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

sys.path.insert(0, os.path.dirname(__file__))
from src.model import load_ensemble
from src.inference import load_image, to_tensor, predict_ensemble, confidence_tier, estimate_severity
from src.gradcam import GradCAM, overlay_heatmap
from src.database import save_prediction, get_history, clear_history
from src.report import generate_pdf
from src.ai_explainer import get_gpt4_explanation, get_lime_fallback

# ─── Config ───────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="PneumoAI — Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; }

/* Card */
.card {
    background: linear-gradient(135deg, #1e2130 0%, #16192a 100%);
    border: 1px solid #2d3250;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Result boxes */
.result-pneumonia {
    background: linear-gradient(135deg, #3d1515 0%, #2d1010 100%);
    border: 2px solid #e53e3e;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.result-normal {
    background: linear-gradient(135deg, #0d2d18 0%, #0a2012 100%);
    border: 2px solid #38a169;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.result-label {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.conf-green  { color: #48bb78; font-weight: 600; }
.conf-yellow { color: #ecc94b; font-weight: 600; }
.conf-red    { color: #fc8181; font-weight: 600; }

/* Severity */
.severity-pill {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 6px;
}
.sev-minimal  { background: #2d6a4f; color: #b7e4c7; }
.sev-mild     { background: #1e4d2b; color: #d8f3dc; }
.sev-moderate { background: #7d4e00; color: #ffe0a3; }
.sev-severe   { background: #7d1a1a; color: #fecaca; }

/* Model bar */
.model-bar {
    background: #1a1d2e;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.88rem;
}

/* Disclaimer */
.disclaimer {
    background: #1a1500;
    border: 1px solid #b7791f;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #fbd38d;
    margin-top: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #12151f; }
</style>
""", unsafe_allow_html=True)

# ─── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    if not os.path.exists(os.path.join(MODELS_DIR, "densenet121_best.pth")):
        return None, None, None, None
    dn, en = load_ensemble(MODELS_DIR, DEVICE)
    dn_cam = GradCAM(dn, "densenet121", device=DEVICE)
    en_cam = GradCAM(en, "efficientnet_b0", device=DEVICE)
    return dn, en, dn_cam, en_cam

@st.cache_data
def load_metrics():
    tm = os.path.join(METRICS_DIR, "test_metrics.json")
    mc = os.path.join(METRICS_DIR, "training_metrics.csv")
    ns = os.path.join(MODELS_DIR, "norm_stats.json")
    metrics = json.load(open(tm)) if os.path.exists(tm) else None
    curves  = pd.read_csv(mc) if os.path.exists(mc) else None
    norm    = json.load(open(ns)) if os.path.exists(ns) else None
    return metrics, curves, norm

densenet, efficientnet, dn_gradcam, en_gradcam = load_models()
test_metrics, training_curves, norm_stats = load_metrics()

THRESHOLD = 0.5
if test_metrics and "ensemble" in test_metrics:
    THRESHOLD = test_metrics["ensemble"].get("optimal_threshold", 0.5)

MODELS_LOADED = densenet is not None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 PneumoAI")
    st.caption("Chest X-Ray Pneumonia Detection")
    st.divider()

    status = "✅ Loaded" if MODELS_LOADED else "⚠️ Not found"
    st.markdown(f"**Models:** {status}")
    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    st.markdown(f"**Threshold:** `{THRESHOLD:.3f}`")
    st.divider()

    st.markdown("**Ensemble Architecture**")
    st.markdown("- DenseNet121 (CheXNet-inspired)")
    st.markdown("- EfficientNet-B0")
    st.markdown("- Soft voting (avg probabilities)")
    st.divider()

    st.markdown("**Dataset**")
    st.markdown("RSNA Pneumonia Detection Challenge 2018")
    st.markdown("~26,684 chest X-rays (DICOM)")

    if test_metrics and "ensemble" in test_metrics:
        st.divider()
        e = test_metrics["ensemble"]
        st.markdown("**Test Set Performance**")
        st.metric("AUC-ROC", f"{e['auc']:.4f}")
        st.metric("Accuracy", f"{e['accuracy']:.1f}%")
        st.metric("Pneumonia F1", f"{e['f1_pneumonia']:.3f}")

    st.divider()
    st.markdown("**AI Explanation (GPT-4o)**")
    if OPENAI_API_KEY:
        st.success("API key loaded from .env ✅")
    else:
        st.warning("No OPENAI_API_KEY in .env\nFalling back to LIME")
    st.divider()
    st.markdown("""
    <div class="disclaimer">
    ⚠️ For educational/demo use only.<br>
    Not for clinical diagnosis.<br>
    Always consult a radiologist.
    </div>
    """, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🔬 Diagnosis", "📊 Model Performance", "📋 History", "📁 Batch"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("## Chest X-Ray Diagnosis")

    if not MODELS_LOADED:
        st.error("⚠️ Model weights not found in `models/` folder. "
                 "Train models on Kaggle and place weights here first.")
        st.stop()

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload chest X-ray (PNG, JPG, JPEG, DCM)",
            type=["png", "jpg", "jpeg", "dcm"],
            key="single_upload"
        )
    with col_opt:
        st.markdown("**Options**")
        run_explanation = st.checkbox("Run AI Explanation", value=True,
            help="GPT-4o Vision if API key set, else LIME fallback")
        run_en_cam = st.checkbox("Show EfficientNet Grad-CAM too", value=False)

    if uploaded:
        # ── Load & preprocess ──
        with st.spinner("Preprocessing image…"):
            gray = load_image(uploaded)
            tensor = to_tensor(gray)

        # ── Predict ──
        with st.spinner("Running ensemble inference…"):
            result = predict_ensemble(tensor, densenet, efficientnet, DEVICE, THRESHOLD)

        # ── Grad-CAM ──
        cam_class = 1  # always explain Pneumonia class
        with st.spinner("Generating Grad-CAM…"):
            dn_cam_map = dn_gradcam.generate(gray, class_idx=cam_class)
            dn_overlay = overlay_heatmap(gray, dn_cam_map)
            severity = estimate_severity(dn_cam_map)
            en_overlay = None
            if run_en_cam:
                en_cam_map = en_gradcam.generate(gray, class_idx=cam_class)
                en_overlay = overlay_heatmap(gray, en_cam_map)

        # ── AI Explanation (GPT-4o → LIME fallback) ──
        lime_overlay = None
        ai_explanation = None
        used_gpt4 = False
        if run_explanation:
            if OPENAI_API_KEY:
                with st.spinner("GPT-4o Vision analyzing X-ray…"):
                    ai_explanation = get_gpt4_explanation(
                        gray, dn_overlay, result, severity, OPENAI_API_KEY)
                    if ai_explanation["success"]:
                        used_gpt4 = True
                    else:
                        st.warning(f"GPT-4o failed ({ai_explanation['error']}). Running LIME fallback…")
                        lime_overlay, ok, err = get_lime_fallback(gray, densenet, efficientnet, DEVICE)
                        if not ok:
                            st.warning(f"LIME also failed: {err}")
            else:
                with st.spinner("Running LIME fallback (~30s)…"):
                    lime_overlay, ok, err = get_lime_fallback(gray, densenet, efficientnet, DEVICE)
                    if not ok:
                        st.warning(f"LIME failed: {err}")

        # ── Result display ──
        tier_color, tier_msg = confidence_tier(result)
        label_class = "result-pneumonia" if result["label"] == "Pneumonia" else "result-normal"
        label_color = "#fc8181" if result["label"] == "Pneumonia" else "#68d391"
        conf_cls = f"conf-{tier_color}"

        col_res, col_detail = st.columns([1, 1])
        with col_res:
            st.markdown(f"""
            <div class="{label_class}">
                <div class="result-label" style="color:{label_color}">
                    {"🔴" if result["label"]=="Pneumonia" else "🟢"} {result["label"]}
                </div>
                <div class="{conf_cls}" style="margin-top:8px">
                    Confidence: {result["confidence"]:.1f}%
                </div>
                <div style="margin-top:6px; font-size:0.82rem; color:#a0aec0">
                    {tier_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown("<br>", unsafe_allow_html=True)
            bar_color = "#e53e3e" if result["label"] == "Pneumonia" else "#38a169"
            fig_bar = go.Figure(go.Bar(
                x=[result["confidence"]], y=[""],
                orientation="h",
                marker_color=bar_color,
                text=[f"{result['confidence']:.1f}%"],
                textposition="inside",
            ))
            fig_bar.update_layout(
                height=60, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False),
                font_color="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_detail:
            # Individual model probabilities
            st.markdown("**Model Breakdown**")
            for name, prob in [("DenseNet121", result["dn_prob"]),
                               ("EfficientNet-B0", result["en_prob"]),
                               ("Ensemble", result["prob_pneumonia"])]:
                bar_w = int(prob * 100)
                st.markdown(f"""
                <div class="model-bar">
                  <span style="color:#a0aec0">{name}</span>
                  <span style="float:right;color:{'#fc8181' if prob>0.5 else '#68d391'}">
                    {prob:.4f}</span>
                  <div style="background:#2d3748;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:{'#fc8181' if prob>0.5 else '#68d391'};
                         width:{bar_w}%;height:6px;border-radius:4px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Severity
            st.markdown("<br>", unsafe_allow_html=True)
            sev_map = {"Minimal / Normal":"sev-minimal","Mild":"sev-mild",
                       "Moderate":"sev-moderate","Severe":"sev-severe"}
            sev_cls = sev_map.get(severity["severity"], "sev-minimal")
            st.markdown(f"""
            <div class="card">
                <b>Severity Assessment</b><br>
                <span class="severity-pill {sev_cls}">{severity['severity']}</span>
                <div style="color:#a0aec0;font-size:0.85rem;margin-top:8px">
                  Affected scan area: <b style="color:white">{severity['affected_pct']:.1f}%</b>
                </div>
                <div style="color:#718096;font-size:0.75rem;margin-top:4px">
                  Based on Grad-CAM activation area analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Images ──
        st.markdown("---")
        st.markdown("### 🔍 Explainability Views")

        n_img_cols = 3 if (run_en_cam or lime_overlay is not None) else 2
        img_cols = st.columns(n_img_cols)

        orig_rgb = np.stack([gray, gray, gray], axis=-1)
        with img_cols[0]:
            st.image(orig_rgb, caption="Original X-Ray", use_container_width=True)
        with img_cols[1]:
            st.image(dn_overlay, caption="Grad-CAM — DenseNet121 Attention",
                     use_container_width=True)
            st.caption("🔴 Red = high attention | 🔵 Blue = low attention")

        if run_en_cam and en_overlay is not None and n_img_cols > 2:
            with img_cols[2]:
                st.image(en_overlay, caption="Grad-CAM — EfficientNet-B0",
                         use_container_width=True)
        elif lime_overlay is not None and n_img_cols > 2:
            with img_cols[2]:
                st.image(lime_overlay, caption="LIME Fallback — Influential Regions",
                         use_container_width=True)
                st.caption("🟢 Green = superpixels driving prediction")

        # ── GPT-4o Explanation Card ──
        if used_gpt4 and ai_explanation and ai_explanation.get("success"):
            st.markdown("---")
            st.markdown("### 🤖 GPT-4o Vision — Clinical Explanation")
            sections = [
                ("🔬 Radiological Observations", ai_explanation.get("findings", "")),
                ("🗺️ Attention Map Analysis",    ai_explanation.get("attention_analysis", "")),
                ("🤝 Model Agreement Assessment", ai_explanation.get("interpretation", "")),
                ("⚠️ Limitations & Caveats",     ai_explanation.get("clinical_note", "")),
            ]
            for title, content in sections:
                if content:
                    st.markdown(f"""
                    <div class="card">
                        <div style="color:#7c8cf8;font-size:0.8rem;font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">
                            {title}
                        </div>
                        <div style="color:#e2e8f0;font-size:0.92rem;line-height:1.6">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.caption("Powered by GPT-4o Vision · For educational purposes only · Not a clinical diagnosis")

        # ── Save to DB ──
        save_prediction(
            uploaded.name, result, severity, THRESHOLD
        )

        # ── PDF Download ──
        st.markdown("---")
        pdf_bytes = generate_pdf(
            filename=uploaded.name,
            result=result,
            severity=severity,
            gray_uint8=gray,
            gradcam_rgb=dn_overlay,
            lime_rgb=lime_overlay,
            threshold=THRESHOLD,
        )
        st.download_button(
            label="📄 Download Diagnostic Report (PDF)",
            data=pdf_bytes,
            file_name=f"pneumoai_report_{uploaded.name.split('.')[0]}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("## Model Performance")

    if test_metrics is None:
        st.warning("No test_metrics.json found in metrics/ folder.")
    else:
        # Metrics summary cards
        e = test_metrics["ensemble"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ensemble AUC", f"{e['auc']:.4f}")
        c2.metric("Accuracy", f"{e['accuracy']:.1f}%")
        c3.metric("Pneumonia Recall", f"{e['recall_pneumonia']:.3f}")
        c4.metric("Pneumonia Precision", f"{e['precision_pneumonia']:.3f}")
        c5.metric("Macro F1", f"{e['macro_f1']:.3f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            # Model comparison bar
            models_data = {
                "DenseNet121": test_metrics["densenet121"],
                "EfficientNet-B0": test_metrics["efficientnet_b0"],
                "Ensemble": test_metrics["ensemble"],
            }
            fig_cmp = go.Figure()
            colors = ["#4299e1", "#ed8936", "#9f7aea"]
            for i, (name, m) in enumerate(models_data.items()):
                fig_cmp.add_trace(go.Bar(
                    name=name,
                    x=["AUC-ROC", "Accuracy (%)"],
                    y=[m["auc"], m["accuracy"]],
                    marker_color=colors[i],
                ))
            fig_cmp.update_layout(
                title="Model Comparison", barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#2d3748"), yaxis=dict(gridcolor="#2d3748"),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        with col_b:
            # Figures from metrics folder
            fig_dir = os.path.join(METRICS_DIR, "figures")
            roc_path = os.path.join(fig_dir, "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curves", use_container_width=True)

        if training_curves is not None:
            st.markdown("---")
            st.markdown("### Training Curves")
            tc1, tc2, tc3 = st.columns(3)
            for col, (y_col, title) in zip(
                [tc1, tc2, tc3],
                [("val_loss", "Validation Loss"),
                 ("val_acc", "Validation Accuracy (%)"),
                 ("val_auc", "Validation AUC-ROC")]
            ):
                fig = go.Figure()
                for model_name, color in [("densenet121","#4299e1"),("efficientnet_b0","#ed8936")]:
                    sub = training_curves[training_curves["model"] == model_name]
                    fig.add_trace(go.Scatter(
                        x=sub["epoch"], y=sub[y_col], name=model_name,
                        line=dict(color=color, width=2), mode="lines"))
                fig.update_layout(
                    title=title, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                    xaxis=dict(title="Epoch", gridcolor="#2d3748"),
                    yaxis=dict(gridcolor="#2d3748"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10,r=10,t=40,b=10), height=280,
                )
                col.plotly_chart(fig, use_container_width=True)

        # Confusion matrix + PR curve images
        col_cm, col_pr = st.columns(2)
        fig_dir = os.path.join(METRICS_DIR, "figures")
        cm_path = os.path.join(fig_dir, "confusion_matrix.png")
        pr_path = os.path.join(fig_dir, "pr_curves.png")
        if os.path.exists(cm_path):
            col_cm.image(cm_path, caption="Confusion Matrix (Optimal Threshold)")
        if os.path.exists(pr_path):
            col_pr.image(pr_path, caption="Precision-Recall Curves")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("## Prediction History")
    col_h1, col_h2 = st.columns([3,1])
    with col_h2:
        if st.button("🗑️ Clear History", type="secondary"):
            clear_history()
            st.rerun()

    df_hist = get_history()
    if df_hist.empty:
        st.info("No predictions yet. Run a diagnosis in the Diagnosis tab.")
    else:
        # Summary metrics
        n_total = len(df_hist)
        n_pneu = (df_hist["label"] == "Pneumonia").sum()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total Scans", n_total)
        h2.metric("Pneumonia Detected", n_pneu)
        h3.metric("Normal", n_total - n_pneu)
        h4.metric("Avg Confidence", f"{df_hist['confidence'].mean():.1f}%")

        # Pie chart
        fig_pie = px.pie(
            df_hist, names="label", title="Label Distribution",
            color_discrete_map={"Normal":"#38a169","Pneumonia":"#e53e3e"},
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)"), height=260
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            df_hist[["timestamp","filename","label","confidence",
                      "prob_pneumonia","severity","affected_pct"]],
            use_container_width=True, hide_index=True,
        )

        csv = df_hist.to_csv(index=False).encode()
        st.download_button("⬇️ Export CSV", csv, "prediction_history.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("## Batch Processing")
    st.caption("Upload multiple X-rays at once. Results shown as a summary table.")

    if not MODELS_LOADED:
        st.error("Model weights not found. Train on Kaggle first.")
    else:
        batch_files = st.file_uploader(
            "Upload multiple X-rays",
            type=["png","jpg","jpeg","dcm"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if batch_files:
            if st.button("▶️ Run Batch Inference", type="primary"):
                results_list = []
                prog = st.progress(0, text="Processing…")

                for i, f in enumerate(batch_files):
                    try:
                        gray = load_image(f)
                        tensor = to_tensor(gray)
                        result = predict_ensemble(tensor, densenet, efficientnet, DEVICE, THRESHOLD)
                        cam_map = dn_gradcam.generate(gray, class_idx=1)
                        sev = estimate_severity(cam_map)
                        tier_color, _ = confidence_tier(result)
                        results_list.append({
                            "File": f.name,
                            "Label": result["label"],
                            "Confidence (%)": result["confidence"],
                            "P(Pneumonia)": result["prob_pneumonia"],
                            "DenseNet121": result["dn_prob"],
                            "EfficientNet-B0": result["en_prob"],
                            "Severity": sev["severity"],
                            "Affected %": sev["affected_pct"],
                        })
                        save_prediction(f.name, result, sev, THRESHOLD)
                    except Exception as ex:
                        results_list.append({"File": f.name, "Label": f"ERROR: {ex}"})
                    prog.progress((i+1)/len(batch_files), text=f"Processing {i+1}/{len(batch_files)}")

                prog.empty()
                df_batch = pd.DataFrame(results_list)
                st.success(f"✅ Processed {len(batch_files)} images")

                # Summary
                if "Label" in df_batch.columns:
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Total", len(df_batch))
                    b2.metric("Pneumonia", (df_batch["Label"]=="Pneumonia").sum())
                    b3.metric("Normal", (df_batch["Label"]=="Normal").sum())

                st.dataframe(df_batch, use_container_width=True, hide_index=True)

                csv_b = df_batch.to_csv(index=False).encode()
                st.download_button("⬇️ Download Batch Results CSV", csv_b,
                                   "batch_results.csv", "text/csv")
