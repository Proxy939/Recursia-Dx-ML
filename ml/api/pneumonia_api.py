"""
Pneumonia Detection API — DenseNet121 + EfficientNet-B0 Ensemble
Binary Classification: Normal | Pneumonia (from chest X-rays)
Serves on port 5003
"""

import os
import sys
import time
import argparse
import logging
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
# The pneumonia_detection package lives at ml/pneumonia_detection/pneumonia_detection/
PNEUMONIA_PKG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'pneumonia_detection', 'pneumonia_detection'
)
MODELS_DIR = os.path.join(PNEUMONIA_PKG_DIR, 'models')

# Add the pneumonia package to sys.path so `from src.* import ...` works
if PNEUMONIA_PKG_DIR not in sys.path:
    sys.path.insert(0, PNEUMONIA_PKG_DIR)

# ── Flask ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Global model references
densenet = None
efficientnet = None
gradcam_dn = None
gradcam_en = None
device = "cpu"
OPTIMAL_THRESHOLD = 0.4579  # From training ROC analysis


def load_models():
    """Load the DenseNet121 + EfficientNet-B0 ensemble."""
    global densenet, efficientnet, gradcam_dn, gradcam_en, device

    try:
        import torch
        from src.model import load_ensemble
        from src.gradcam import GradCAM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        dn_path = os.path.join(MODELS_DIR, 'densenet121_best.pth')
        en_path = os.path.join(MODELS_DIR, 'efficientnet_b0_best.pth')

        if not os.path.exists(dn_path):
            logger.warning(f"DenseNet121 weights not found: {dn_path}")
            return False
        if not os.path.exists(en_path):
            logger.warning(f"EfficientNet-B0 weights not found: {en_path}")
            return False

        logger.info(f"Loading ensemble from: {MODELS_DIR}")
        densenet, efficientnet = load_ensemble(MODELS_DIR, device)

        # Initialize Grad-CAM for both models
        gradcam_dn = GradCAM(densenet, "densenet121", device)
        gradcam_en = GradCAM(efficientnet, "efficientnet_b0", device)

        logger.info("Pneumonia ensemble loaded successfully")
        logger.info(f"  DenseNet121:    {sum(p.numel() for p in densenet.parameters())/1e6:.1f}M params")
        logger.info(f"  EfficientNet-B0: {sum(p.numel() for p in efficientnet.parameters())/1e6:.1f}M params")
        logger.info(f"  Optimal threshold: {OPTIMAL_THRESHOLD}")
        return True

    except Exception as e:
        logger.error(f"Failed to load pneumonia models: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': densenet is not None and efficientnet is not None,
        'model_type': 'DenseNet121 + EfficientNet-B0 Ensemble',
        'task': 'Pneumonia Detection from Chest X-rays',
        'classes': ['Normal', 'Pneumonia'],
        'num_classes': 2,
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'device': device,
        'timestamp': time.time()
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a chest X-ray image for pneumonia.

    Accepts: POST with multipart/form-data containing 'image' file
    Returns: JSON with prediction, confidence, severity, per-model probabilities, heatmap
    """
    global densenet, efficientnet, gradcam_dn

    try:
        if densenet is None or efficientnet is None:
            return jsonify({
                'success': False,
                'error': 'Pneumonia models not loaded'
            }), 503

        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided. Send image as multipart/form-data with key "image".'
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Import inference utilities
        from src.inference import load_image, to_tensor, predict_ensemble, confidence_tier, estimate_severity
        from src.gradcam import overlay_heatmap

        start_time = time.time()

        # Load and preprocess
        gray_img = load_image(file)
        tensor = to_tensor(gray_img)

        # Run ensemble prediction
        result = predict_ensemble(tensor, densenet, efficientnet, device, threshold=OPTIMAL_THRESHOLD)

        # Generate Grad-CAM heatmap (use DenseNet121 — larger receptive field)
        import torch
        torch.set_grad_enabled(True)
        cam = gradcam_dn.generate(gray_img, class_idx=1)  # class 1 = Pneumonia
        torch.set_grad_enabled(False)

        # Estimate severity from heatmap
        severity_info = estimate_severity(cam)

        # Create heatmap overlay image and encode as base64
        overlay = overlay_heatmap(gray_img, cam)
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format='PNG')
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Confidence tier
        tier_color, tier_message = confidence_tier(result)

        processing_time = time.time() - start_time

        response = {
            'success': True,
            'predicted_class': result['label'],
            'confidence_percent': result['confidence'],
            'is_pneumonia': result['label'] == 'Pneumonia',
            'prob_pneumonia': result['prob_pneumonia'],
            'per_model': {
                'densenet121': result['dn_prob'],
                'efficientnet_b0': result['en_prob'],
                'ensemble': result['prob_pneumonia']
            },
            'severity': severity_info['severity'],
            'affected_area_pct': severity_info['affected_pct'],
            'confidence_tier': {
                'color': tier_color,
                'message': tier_message
            },
            'risk_level': 'High Risk' if result['label'] == 'Pneumonia' and result['confidence'] >= 70 else
                          'Moderate Risk' if result['label'] == 'Pneumonia' else 'Low Risk',
            'heatmap_base64': heatmap_b64,
            'metadata': {
                'model': 'DenseNet121 + EfficientNet-B0 Ensemble',
                'input_size': 224,
                'threshold': OPTIMAL_THRESHOLD,
                'processing_time_ms': round(processing_time * 1000, 1),
                'device': device,
                'dataset': 'RSNA Pneumonia Detection Challenge 2018',
                'auc': 0.8889
            }
        }

        logger.info(f"Prediction: {result['label']} ({result['confidence']:.2f}%) "
                     f"severity={severity_info['severity']} in {processing_time:.3f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information."""
    return jsonify({
        'success': True,
        'model_type': 'DenseNet121 + EfficientNet-B0 Ensemble (Soft Voting)',
        'task': 'Pneumonia Detection from Chest X-rays (Binary)',
        'classes': ['Normal', 'Pneumonia'],
        'input_size': '224x224',
        'framework': 'PyTorch',
        'pretrained_on': 'ImageNet',
        'dataset': 'RSNA Pneumonia Detection Challenge 2018 (~26,684 DICOM images)',
        'ensemble_method': 'Soft voting (average softmax probabilities)',
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'metrics': {
            'auc_roc': 0.8889,
            'accuracy': '77.89% (optimal threshold)',
            'recall_pneumonia': 0.86,
            'precision_pneumonia': 0.51,
            'f1_pneumonia': 0.64,
            'macro_f1': 0.74
        },
        'individual_models': {
            'densenet121': {'auc': 0.8798, 'accuracy': '74.27%', 'params': '~8M'},
            'efficientnet_b0': {'auc': 0.8853, 'accuracy': '81.19%', 'params': '~5.3M'}
        },
        'features': [
            'Ensemble inference',
            'Grad-CAM heatmap',
            'Severity scoring',
            'Confidence tiering',
            'DICOM file support'
        ],
        'model_loaded': densenet is not None and efficientnet is not None
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pneumonia Detection API')
    parser.add_argument('--port', type=int, default=5003, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  PneumoAI — Pneumonia Detection API")
    logger.info("  DenseNet121 + EfficientNet-B0 Ensemble")
    logger.info("=" * 60)

    if load_models():
        logger.info("Model ready for inference")
    else:
        logger.warning("Starting without models - /analyze will return 503")

    logger.info(f"Starting Pneumonia API on port {args.port}...")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
