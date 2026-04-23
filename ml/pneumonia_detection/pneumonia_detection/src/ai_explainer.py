"""
ai_explainer.py — GPT-4o Vision explanation for X-ray predictions.
Falls back to LIME if OpenAI API is unavailable.
"""

import base64
import io
import numpy as np
from PIL import Image


def _encode_image_b64(arr_rgb: np.ndarray) -> str:
    """Convert RGB uint8 numpy array to base64 PNG string."""
    pil = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_gpt4_explanation(
    gray_uint8: np.ndarray,
    gradcam_rgb: np.ndarray,
    result: dict,
    severity: dict,
    api_key: str,
) -> dict:
    """
    Send the original X-ray + Grad-CAM overlay to GPT-4o Vision.
    Returns dict with keys: findings, interpretation, attention_analysis,
    clinical_note, success (bool), error (str|None)
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"success": False, "error": "openai package not installed. Run: pip install openai"}

    try:
        client = OpenAI(api_key=api_key)

        orig_rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
        orig_b64 = _encode_image_b64(orig_rgb)
        cam_b64  = _encode_image_b64(gradcam_rgb)

        label      = result["label"]
        confidence = result["confidence"]
        dn_prob    = result["dn_prob"]
        en_prob    = result["en_prob"]
        ens_prob   = result["prob_pneumonia"]
        severity_label = severity.get("severity", "Unknown")
        affected_pct   = severity.get("affected_pct", 0.0)

        system_prompt = (
            "You are an expert radiologist assistant reviewing chest X-rays. "
            "You will be shown two images: the original chest X-ray and a Grad-CAM "
            "attention heatmap (red=high model attention, blue=low). "
            "You MUST base your analysis strictly on what you observe in the images. "
            "Do NOT invent findings. Be concise, specific, and clinically relevant. "
            "Do NOT provide a clinical diagnosis — this is an educational AI tool."
        )

        user_prompt = f"""An AI ensemble (DenseNet121 + EfficientNet-B0) has analyzed this chest X-ray.

Model Output:
- Prediction: {label}
- Ensemble Probability of Pneumonia: {ens_prob:.4f} ({confidence:.1f}% confidence)
- DenseNet121 probability: {dn_prob:.4f}
- EfficientNet-B0 probability: {en_prob:.4f}
- Estimated severity: {severity_label} ({affected_pct:.1f}% scan area highlighted)

Image 1: Original chest X-ray
Image 2: Grad-CAM attention heatmap (red = areas model focused on most)

Please provide:

1. **Radiological Observations**: What specific features do you observe in the chest X-ray? (e.g. lung fields, opacity, consolidation, infiltrates, pleural effusion, cardiac silhouette)

2. **Attention Map Analysis**: Which anatomical regions are highlighted in the Grad-CAM? Does the attention align with expected radiological findings for {label}?

3. **Model Agreement Assessment**: Do the highlighted regions correspond to areas that would be clinically relevant for pneumonia detection? Are both models likely focusing on the same regions (given their similar/different probabilities)?

4. **Limitations & Caveats**: What should a radiologist be cautious about when interpreting this AI output?

Keep each section to 2-3 sentences. Be specific about image regions (e.g. "right lower lobe", "perihilar region")."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{orig_b64}",
                                       "detail": "high"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{cam_b64}",
                                       "detail": "high"}},
                    ]
                }
            ],
            max_tokens=700,
            temperature=0.2,  # low temp for more factual, consistent output
        )

        text = response.choices[0].message.content
        # Parse the markdown sections
        sections = {}
        current_key = None
        current_lines = []
        key_map = {
            "radiological observations": "findings",
            "attention map analysis":    "attention_analysis",
            "model agreement assessment":"interpretation",
            "limitations & caveats":     "clinical_note",
        }
        for line in text.split("\n"):
            stripped = line.strip()
            matched = False
            for header, key in key_map.items():
                if header in stripped.lower():
                    if current_key and current_lines:
                        sections[current_key] = " ".join(current_lines).strip()
                    current_key = key
                    current_lines = []
                    matched = True
                    break
            if not matched and current_key and stripped:
                # Remove markdown bold markers
                clean = stripped.lstrip("*#- ").strip()
                if clean:
                    current_lines.append(clean)

        if current_key and current_lines:
            sections[current_key] = " ".join(current_lines).strip()

        # Fallback: if parsing failed, return raw text
        if not sections:
            sections = {"findings": text}

        sections["success"] = True
        sections["error"]   = None
        sections["raw"]     = text
        return sections

    except Exception as e:
        return {"success": False, "error": str(e), "raw": ""}


def get_lime_fallback(
    gray_uint8: np.ndarray,
    densenet,
    efficientnet,
    device: str,
) -> tuple:
    """
    Run LIME as fallback. Returns (lime_overlay_rgb, success, error_msg).
    """
    try:
        from src.lime_explainer import run_lime, overlay_lime
        mask = run_lime(gray_uint8, densenet, efficientnet, device,
                        num_samples=300, num_features=6)
        overlay = overlay_lime(gray_uint8, mask)
        return overlay, True, None
    except Exception as e:
        return None, False, str(e)
