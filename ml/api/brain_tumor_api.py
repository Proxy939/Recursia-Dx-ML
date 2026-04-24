"""
Brain Tumor EfficientNetB3 API
4-Class MRI Classification: Glioma | Meningioma | Pituitary | No Tumor
Includes Grad-CAM heatmap explainability (GradientTape).
Serves on port 5002
"""

import os
import io
import time
import base64
import argparse
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE    = 300
CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')
MODEL_PATH  = os.path.join(WEIGHTS_DIR, 'brain_tumor_efficientnetb3.h5')

app = Flask(__name__)
CORS(app)

model      = None
grad_model = None   # Separate model for Grad-CAM (shares weights, no copy)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    global model, grad_model
    try:
        import tensorflow as tf

        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model not found: {MODEL_PATH}")
            return False

        logger.info(f"Loading brain tumor model: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded | input={model.input_shape} | output={model.output_shape}")

        # ── Find best Grad-CAM target layer ────────────────────────────
        # We want the last Conv/BatchNorm layer with a 4-D spatial output
        # before global average pooling. Skip pooling, dropout, reshape layers.
        SKIP_KEYWORDS = ('global', 'pool', 'dropout', 'flatten', 'reshape',
                         'padding', 'input', 'lambda')
        target_layer = None
        for layer in reversed(model.layers):
            name_lower = layer.name.lower()
            if any(kw in name_lower for kw in SKIP_KEYWORDS):
                continue
            try:
                out_shape = layer.output_shape
                # Need 4-D: (batch, H, W, C) — H and W must be > 1
                if (isinstance(out_shape, (list, tuple)) and
                        len(out_shape) == 4 and
                        out_shape[1] is not None and out_shape[1] > 1 and
                        out_shape[2] is not None and out_shape[2] > 1):
                    target_layer = layer
                    break
            except Exception:
                continue

        if target_layer:
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )
            logger.info(f"✅ Grad-CAM target layer: '{target_layer.name}' "
                        f"| output shape: {target_layer.output_shape}")
        else:
            logger.warning("⚠️  No suitable Grad-CAM layer found — heatmaps disabled")
            grad_model = None

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        grad_model = None
        return False


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(pil_image):
    import tensorflow as tf
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(pil_image, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def generate_gradcam(img_array, class_idx):
    """
    Compute Grad-CAM using TensorFlow GradientTape.
    Explicitly watches the intermediate conv-layer output (not just the input).
    Returns float32 ndarray [0,1] of shape (IMG_SIZE, IMG_SIZE), or None.
    """
    import tensorflow as tf

    if grad_model is None:
        logger.warning("Grad-CAM skipped: grad_model is None")
        return None

    try:
        img_tensor = tf.cast(img_array, tf.float32)

        # ── KEY FIX: watch the intermediate layer output, not just the input ──
        with tf.GradientTape() as tape:
            # Run forward pass; grad_model returns [conv_output, final_predictions]
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            # Tell tape to track the intermediate conv tensor explicitly
            tape.watch(conv_outputs)
            loss = predictions[:, class_idx]

        # Gradient of class score w.r.t. feature maps
        grads = tape.gradient(loss, conv_outputs)  # (1, H, W, C)

        if grads is None:
            logger.warning("Grad-CAM: tape.gradient returned None — "
                           "ensure the model was NOT built with @tf.function tracing issues")
            return None

        # Global average pool over spatial dims → importance weights (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by their importance
        conv_out_np   = conv_outputs[0].numpy()       # (H, W, C)
        pooled_np     = pooled_grads.numpy()          # (C,)
        cam = np.dot(conv_out_np, pooled_np)          # (H, W)

        # ReLU + normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        else:
            logger.warning("Grad-CAM: activation map is all zeros — "
                           "model may be predicting with near-zero gradient for this class")
            return None

        # Resize to model input resolution
        try:
            import cv2
            cam = cv2.resize(cam.astype(np.float32), (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback: PIL-based resize
            from PIL import Image as PILImage
            cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
            cam = np.array(cam_pil).astype(np.float32) / 255.0

        logger.info(f"Grad-CAM OK | cam min={cam.min():.4f} max={cam.max():.4f} "
                    f"| active_px={int(np.mean(cam > 0.5) * 100)}%")
        return cam.astype(np.float32)

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
        return None


def overlay_heatmap(original_pil, cam, alpha=0.45):
    """Blend jet-colormap Grad-CAM over the original image. Returns base64 PNG."""
    try:
        import cv2
        orig_rgb = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))
        heat_u8  = (cam * 255).astype(np.uint8)
        cmap_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB)
        blended  = cv2.addWeighted(orig_rgb, 1 - alpha, cmap_rgb, alpha, 0)
        result_img = Image.fromarray(blended)
    except ImportError:
        # Pure-PIL fallback colorizer (matplotlib jet approximation)
        import colorsys
        orig_rgb   = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))
        h, w       = cam.shape
        cmap_rgb   = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                v = float(cam[y, x])
                # Jet colormap approximation: blue→cyan→green→yellow→red
                r = int(np.clip(1.5 - abs(4 * v - 3), 0, 1) * 255)
                g = int(np.clip(1.5 - abs(4 * v - 2), 0, 1) * 255)
                b = int(np.clip(1.5 - abs(4 * v - 1), 0, 1) * 255)
                cmap_rgb[y, x] = [r, g, b]
        blended = (orig_rgb * (1 - alpha) + cmap_rgb * alpha).astype(np.uint8)
        result_img = Image.fromarray(blended)

    buf = io.BytesIO()
    result_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def estimate_severity(cam, threshold=0.5):
    """Derive severity label and affected-area % from Grad-CAM activation."""
    pct = round(float(np.mean(cam > threshold)) * 100, 1)
    severity = ("Minimal" if pct < 10 else
                "Mild"    if pct < 20 else
                "Moderate" if pct < 35 else
                "Severe")
    return severity, pct


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gradcam_enabled': grad_model is not None,
        'model_type': 'EfficientNetB3',
        'task': 'Brain Tumor MRI Classification',
        'classes': CLASSES,
        'num_classes': NUM_CLASSES,
        'timestamp': time.time()
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Classify a brain MRI and return Grad-CAM heatmap.

    Input : multipart/form-data  { image: <file> }
    Output: predicted_class, confidence_percent, all_class_probabilities,
            is_tumor, risk_level,
            heatmap_base64, severity, affected_area_pct  (Grad-CAM fields)
    """
    global model, grad_model

    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Brain tumor model not loaded',
                'message': 'Run train_brain_tumor.py first.'
            }), 503

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        # ── Load image ───────────────────────────────────────────────────
        t0 = time.time()
        image_bytes = file.read()
        original_pil = Image.open(io.BytesIO(image_bytes))
        img_array = preprocess_image(original_pil)

        # ── Predict ──────────────────────────────────────────────────────
        probs_raw      = model.predict(img_array, verbose=0)[0]
        predicted_idx  = int(np.argmax(probs_raw))
        predicted_class = CLASSES[predicted_idx]
        confidence      = float(probs_raw[predicted_idx])
        all_probs       = {cls: round(float(probs_raw[i]) * 100, 2)
                           for i, cls in enumerate(CLASSES)}

        # ── Grad-CAM ─────────────────────────────────────────────────────
        heatmap_b64  = None
        severity     = None
        affected_pct = None

        cam = generate_gradcam(img_array, predicted_idx)
        if cam is not None:
            heatmap_b64  = overlay_heatmap(original_pil, cam)
            severity, affected_pct = estimate_severity(cam)
            logger.info(f"Grad-CAM OK | severity={severity} | affected={affected_pct}%")
        else:
            logger.info("Grad-CAM skipped (not available)")

        elapsed = time.time() - t0
        logger.info(f"Prediction: {predicted_class} ({confidence:.4f}) in {elapsed:.3f}s")

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence_percent': round(confidence * 100, 2),
            'all_class_probabilities': all_probs,
            'is_tumor': predicted_class != 'notumor',
            'risk_level': (
                'High Risk'     if predicted_class != 'notumor' and confidence > 0.7 else
                'Moderate Risk' if predicted_class != 'notumor' else
                'Low Risk'
            ),
            # ── Grad-CAM output (same field names as pneumonia API) ──────
            'heatmap_base64': heatmap_b64,
            'severity': severity,
            'affected_area_pct': affected_pct,
            'gradcam_enabled': cam is not None,
            # ── Metadata ─────────────────────────────────────────────────
            'metadata': {
                'model': 'EfficientNetB3',
                'input_size': IMG_SIZE,
                'processing_time_ms': round(elapsed * 1000, 1),
                'classes': CLASSES,
                'prediction': {
                    'predicted_class': predicted_class,
                    'probabilities': all_probs
                }
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'success': True,
        'model_type': 'EfficientNetB3',
        'task': 'Brain Tumor MRI Classification (4-class)',
        'classes': CLASSES,
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'framework': 'TensorFlow/Keras',
        'pretrained_on': 'ImageNet',
        'dataset': 'masoudnickparvar/brain-tumor-mri-dataset',
        'gradcam_enabled': grad_model is not None,
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain Tumor EfficientNetB3 API  (+ Grad-CAM)')
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Brain Tumor EfficientNetB3 API  (+ Grad-CAM)")
    logger.info("=" * 60)

    if load_model():
        status = "Grad-CAM ENABLED" if grad_model else "Grad-CAM DISABLED"
        logger.info(f"Model ready | {status}")
    else:
        logger.warning("Starting without model — /analyze returns 503")

    logger.info(f"Starting on port {args.port}...")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
