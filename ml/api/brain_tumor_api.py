"""
Brain Tumor EfficientNetB3 API
4-Class MRI Classification: Glioma | Meningioma | Pituitary | No Tumor
Serves on port 5002
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = 300  # EfficientNetB3 native input size
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4

# Model path
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'brain_tumor_efficientnetb3.h5')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model reference
model = None


def load_model():
    """Load the EfficientNetB3 brain tumor model."""
    global model
    try:
        import tensorflow as tf
        
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"⚠️ Model file not found at: {MODEL_PATH}")
            logger.warning("  Run 'python train_brain_tumor.py' to train and save the model.")
            return False
        
        logger.info(f"Loading brain tumor model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Brain tumor EfficientNetB3 model loaded successfully")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        logger.info(f"   Classes: {CLASSES}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load brain tumor model: {e}")
        model = None
        return False


def preprocess_image(image):
    """Preprocess image for EfficientNetB3 prediction."""
    import tensorflow as tf
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply EfficientNet preprocessing (scale to [0, 255] range as expected)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'EfficientNetB3',
        'task': 'Brain Tumor MRI Classification',
        'classes': CLASSES,
        'num_classes': NUM_CLASSES,
        'timestamp': time.time()
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a brain MRI image for tumor classification.
    
    Accepts: POST with multipart/form-data containing 'image' file
    Returns: JSON with predicted_class, confidence_percent, all_class_probabilities
    """
    global model
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Brain tumor model not loaded',
                'message': 'Run train_brain_tumor.py to train and save the model first.'
            }), 503
        
        # Check if image file is present
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
        
        # Load and preprocess image
        start_time = time.time()
        image = Image.open(file.stream)
        img_array = preprocess_image(image)
        
        # Run prediction
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Build all class probabilities
        all_probs = {cls: round(float(probabilities[i]) * 100, 2) for i, cls in enumerate(CLASSES)}
        
        processing_time = time.time() - start_time
        
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence_percent': round(confidence * 100, 2),
            'all_class_probabilities': all_probs,
            'is_tumor': predicted_class != 'notumor',
            'risk_level': 'High Risk' if predicted_class != 'notumor' and confidence > 0.7 else
                          'Moderate Risk' if predicted_class != 'notumor' else 'Low Risk',
            'metadata': {
                'model': 'EfficientNetB3',
                'input_size': IMG_SIZE,
                'processing_time_ms': round(processing_time * 1000, 1),
                'classes': CLASSES
            }
        }
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.4f}) in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information."""
    return jsonify({
        'success': True,
        'model_type': 'EfficientNetB3',
        'task': 'Brain Tumor MRI Classification (4-class)',
        'classes': CLASSES,
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'framework': 'TensorFlow/Keras',
        'pretrained_on': 'ImageNet',
        'dataset': 'masoudnickparvar/brain-tumor-mri-dataset',
        'expected_accuracy': '98%+',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Brain Tumor EfficientNetB3 API')
    parser.add_argument('--port', type=int, default=5002, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  Brain Tumor EfficientNetB3 API")
    logger.info("=" * 60)
    
    # Load model
    if load_model():
        logger.info("✅ Model ready for inference")
    else:
        logger.warning("⚠️ Starting without model - /analyze will return 503")
    
    logger.info(f"🚀 Starting Brain Tumor API on port {args.port}...")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
