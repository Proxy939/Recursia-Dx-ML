"""
report.py — PDF report generation using fpdf2.
"""

import io
import os
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image
from fpdf import FPDF


class DiagnosisReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 58, 95)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "AI Pneumonia Detection System - Diagnostic Report",
                  align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6,
                  "DISCLAIMER: This tool is for educational/demonstration purposes only. "
                  "Not for clinical use. Always consult a qualified radiologist.",
                  align="C")


def _pil_to_tmp(pil_img: Image.Image, suffix=".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_img.save(tmp.name)
    tmp.close()
    return tmp.name


def generate_pdf(
    filename: str,
    result: dict,
    severity: dict,
    gray_uint8: np.ndarray,
    gradcam_rgb: np.ndarray,
    lime_rgb: np.ndarray = None,
    threshold: float = 0.5,
) -> bytes:
    """
    Generate PDF report. Returns raw bytes for st.download_button.
    """
    pdf = DiagnosisReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Metadata ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 8, "Report Details", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"File:        {filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Threshold:   {threshold:.3f} (optimal from ROC curve)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Prediction Result ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 8, "Prediction Result", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 18)
    color = (220, 50, 50) if result["label"] == "Pneumonia" else (34, 139, 34)
    pdf.set_text_color(*color)
    pdf.cell(0, 12, f"  {result['label']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  Confidence:       {result['confidence']:.1f}%",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  P(Pneumonia):     {result['prob_pneumonia']:.4f}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  DenseNet121:      {result['dn_prob']:.4f}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  EfficientNet-B0:  {result['en_prob']:.4f}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # --- Severity ---
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 8, "Severity Assessment (Grad-CAM Area Analysis)",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  Severity Level:   {severity.get('severity', 'N/A')}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Affected Area:    {severity.get('affected_pct', 0):.1f}% of scan",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Images ---
    tmp_files = []
    try:
        orig_pil = Image.fromarray(gray_uint8, mode="L").convert("RGB")
        tmp_orig = _pil_to_tmp(orig_pil)
        tmp_files.append(tmp_orig)

        tmp_cam = _pil_to_tmp(Image.fromarray(gradcam_rgb))
        tmp_files.append(tmp_cam)

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 245, 255)
        pdf.cell(0, 8, "Images", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        w = 82
        x1, x2 = 15, 108
        y_img = pdf.get_y()
        pdf.image(tmp_orig, x=x1, y=y_img, w=w)
        pdf.image(tmp_cam, x=x2, y=y_img, w=w)
        pdf.set_y(y_img + w + 2)

        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(w + 4, 5, "    Original X-Ray", new_x="RIGHT")
        pdf.cell(w + 4, 5, "    Grad-CAM Attention Map", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        if lime_rgb is not None:
            tmp_lime = _pil_to_tmp(Image.fromarray(lime_rgb))
            tmp_files.append(tmp_lime)
            y_img2 = pdf.get_y()
            pdf.image(tmp_lime, x=x1, y=y_img2, w=w)
            pdf.set_y(y_img2 + w + 2)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, "    LIME Superpixel Explanation (green = influential regions)",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)

    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    # --- Disclaimer box ---
    pdf.ln(5)
    pdf.set_fill_color(255, 243, 205)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "  IMPORTANT DISCLAIMER", fill=True,
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    disclaimers = [
        "This tool is built for educational and demonstration purposes only.",
        "It is NOT intended for clinical use or actual medical diagnosis.",
        "Always consult a qualified radiologist for medical imaging interpretation.",
        "Model predictions may be inaccurate and should not be acted upon clinically.",
    ]
    for d in disclaimers:
        pdf.cell(0, 5, f"  * {d}", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
