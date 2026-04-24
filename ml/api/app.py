from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import logging
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
from PIL import Image
import io
import base64
import traceback

# ===================================================
# 🔒 DETERMINISTIC SETUP - MUST BE BEFORE TORCH IMPORT
# ===================================================
import random
random.seed(42)

import numpy as np
np.random.seed(42)

# Set environment variables for deterministic behavior
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import our ML modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: Malaria/Platelet models removed. Only Brain Tumor + Pneumonia remain.
from utils.image_utils import validate_image_format, get_image_info, enhance_medical_image
from utils.data_manager import DataManager, save_prediction_report, validate_prediction_data
from config.config import get_config, ERROR_MESSAGES
import requests as http_requests  # For proxying to Brain Tumor API

# Brain Tumor API configuration
BRAIN_TUMOR_API_URL = os.environ.get('BRAIN_TUMOR_API_URL', 'http://localhost:5002')

# Pneumonia Detection API configuration
PNEUMONIA_API_URL = os.environ.get('PNEUMONIA_API_URL', 'http://localhost:5003')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_config(os.getenv('FLASK_ENV', 'development'))
config.create_directories()

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOADS_DIR

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'ml_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components (brain tumor → port 5002, pneumonia → port 5003)
pipeline = None
data_manager = DataManager(str(config.DATABASE_PATH))

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # NumPy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # NumPy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def initialize_models():
    """Initialize ML model proxies for brain tumor and pneumonia detection."""
    global pipeline
    try:
        # ===================================================
        # 🔒 PYTORCH DETERMINISTIC SETUP
        # ===================================================
        import torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)  # Disable gradients for inference
        
        # Use deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
            logger.info("✅ PyTorch deterministic algorithms enabled")
        except Exception as det_error:
            logger.warning(f"⚠️ Could not enable deterministic algorithms: {det_error}")
        
        # ===================================================
        # BRAIN TUMOR DETECTION: Handled by Brain Tumor API (port 5002)
        # ===================================================
        logger.info(f"🔗 Brain tumor analysis will be proxied to Brain Tumor API at: {BRAIN_TUMOR_API_URL}")
        try:
            bt_health = http_requests.get(f"{BRAIN_TUMOR_API_URL}/health", timeout=2)
            if bt_health.status_code == 200:
                logger.info("✅ Brain Tumor API is available")
            else:
                logger.warning("⚠️ Brain Tumor API returned non-200 status")
        except Exception:
            logger.warning("⚠️ Brain Tumor API not reachable - brain tumor analysis may fail")
            logger.warning(f"   Start Brain Tumor API: python api/brain_tumor_api.py --port 5002")
        
        # ===================================================
        # PNEUMONIA DETECTION: Handled by Pneumonia API (port 5003)
        # ===================================================
        logger.info(f"🔗 Pneumonia analysis will be proxied to Pneumonia API at: {PNEUMONIA_API_URL}")
        try:
            pn_health = http_requests.get(f"{PNEUMONIA_API_URL}/health", timeout=2)
            if pn_health.status_code == 200:
                logger.info("✅ Pneumonia API is available")
            else:
                logger.warning("⚠️ Pneumonia API returned non-200 status")
        except Exception:
            logger.warning("⚠️ Pneumonia API not reachable - pneumonia analysis may fail")
            logger.warning(f"   Start Pneumonia API: python api/pneumonia_api.py --port 5003")
        
        logger.info("="*70)
        logger.info("✅ Server initialization complete")
        logger.info(f"   - Brain Tumor Detection: Proxied to Brain Tumor API ({BRAIN_TUMOR_API_URL})")
        logger.info(f"   - Pneumonia Detection:  Proxied to Pneumonia API ({PNEUMONIA_API_URL})")
        logger.info("="*70)
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': ERROR_MESSAGES['image_too_large']
    }), 413

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': ERROR_MESSAGES['unknown_error']
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global pipeline
    
    # Check Brain Tumor API status
    brain_tumor_available = False
    try:
        response = http_requests.get(f"{BRAIN_TUMOR_API_URL}/health", timeout=2)
        brain_tumor_available = response.status_code == 200
    except:
        pass
    
    # Check Pneumonia API status
    pneumonia_available = False
    try:
        response = http_requests.get(f"{PNEUMONIA_API_URL}/health", timeout=2)
        pneumonia_available = response.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'models': {
            'brain_tumor_detection': brain_tumor_available,
            'pneumonia_detection': pneumonia_available,
        },
        'brain_tumor_api': BRAIN_TUMOR_API_URL,
        'pneumonia_api': PNEUMONIA_API_URL,
        'pipeline_loaded': pipeline is not None,
        'timestamp': time.time()
    })

@app.route('/convert-dicom', methods=['POST'])
def convert_dicom():
    """Convert a DICOM (.dcm) file to PNG for browser preview."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        fname = file.filename or ''

        if not fname.lower().endswith('.dcm'):
            return jsonify({'success': False, 'error': 'File is not a DICOM file'}), 400

        import pydicom
        import cv2

        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > 0:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # Apply CLAHE for better visualization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)

        # Convert grayscale to RGB for PNG
        pil_img = Image.fromarray(arr, mode='L').convert('RGB')
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)

        return send_file(buf, mimetype='image/png', download_name=fname.replace('.dcm', '.png'))

    except Exception as e:
        logger.error(f"DICOM conversion error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_tumor():
    """Predict from uploaded image with routing based on imageType."""

    
    try:
        # Get imageType parameter to route to correct model
        image_type = request.form.get('imageType', 'tissue').lower()
        
        logger.info(f"📋 Single predict request for imageType: {image_type}")
        
        # Route to appropriate model
        if image_type == 'tissue':
            # Brain tumor analysis handled by Brain Tumor API (port 5002)
            model_name = "Brain Tumor Detection (EfficientNetB3)"
        elif image_type == 'pneumonia' or image_type == 'lung' or image_type == 'xray':
            # Pneumonia detection handled by Pneumonia API (port 5003)
            image_type = 'pneumonia'  # normalize
            model_name = "Pneumonia Detection (DenseNet121 + EfficientNet-B0 Ensemble)"
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid imageType: {image_type}. Must be "tissue" or "pneumonia"'
            }), 400
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['invalid_image_format']
            }), 400
        
        # Get optional parameters
        user_id = request.form.get('user_id')
        enhance_image = request.form.get('enhance_image', 'false').lower() == 'true'
        save_result = request.form.get('save_result', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOADS_DIR, filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        start_time = time.time()
        
        try:
            # Make prediction based on image type
            if image_type == 'tissue':
                # Load and preprocess image (PIL — works for standard image formats)
                image = Image.open(filepath)
                image_array = np.array(image.convert('RGB'))
                
                # Apply enhancement if requested
                if enhance_image:
                    image_array = enhance_medical_image(image_array)
                
                # Proxy to Brain Tumor API for MRI analysis
                try:
                    with open(filepath, 'rb') as img_file:
                        files = {'image': (filename, img_file, 'image/png')}
                        bt_response = http_requests.post(
                            f"{BRAIN_TUMOR_API_URL}/analyze",
                            files=files,
                            timeout=60
                        )
                    
                    if bt_response.status_code == 200:
                        bt_data = bt_response.json()
                        if bt_data.get('success'):
                            prediction_result = {
                                'predicted_class': bt_data['predicted_class'],
                                'confidence': bt_data['confidence_percent'] / 100.0,
                                'is_tumor': bt_data['predicted_class'] != 'notumor',
                                'probabilities': bt_data.get('all_class_probabilities', {}),
                                'risk_level': 'High Risk' if bt_data['predicted_class'] != 'notumor' else 'Low Risk',
                                'risk_assessment': 'high' if bt_data['predicted_class'] != 'notumor' else 'low',
                                # Grad-CAM fields — pass through from brain_tumor_api
                                'heatmap_base64': bt_data.get('heatmap_base64'),
                                'severity': bt_data.get('severity'),
                                'affected_area_pct': bt_data.get('affected_area_pct'),
                                'gradcam_enabled': bt_data.get('gradcam_enabled', False)
                            }
                        else:
                            raise Exception(bt_data.get('error', 'Brain Tumor API prediction failed'))
                    else:
                        raise Exception(f"Brain Tumor API returned status {bt_response.status_code}")
                        
                except http_requests.exceptions.ConnectionError:
                    return jsonify({
                        'success': False,
                        'error': 'Brain Tumor API is not available',
                        'message': 'Please start the Brain Tumor API: python api/brain_tumor_api.py --port 5002'
                    }), 503
                except http_requests.exceptions.Timeout:
                    return jsonify({
                        'success': False,
                        'error': 'Brain Tumor API request timed out'
                    }), 504

            elif image_type == 'pneumonia':
                # Proxy to Pneumonia API for chest X-ray analysis
                try:
                    # Detect correct MIME type (DICOM vs standard image)
                    is_dicom = filename.lower().endswith('.dcm')
                    mime_type = 'application/dicom' if is_dicom else 'image/png'
                    with open(filepath, 'rb') as img_file:
                        files = {'image': (filename, img_file, mime_type)}
                        pn_response = http_requests.post(
                            f"{PNEUMONIA_API_URL}/analyze",
                            files=files,
                            timeout=60
                        )
                    
                    if pn_response.status_code == 200:
                        pn_data = pn_response.json()
                        if pn_data.get('success'):
                            prediction_result = {
                                'predicted_class': pn_data['predicted_class'],
                                'confidence': pn_data['confidence_percent'] / 100.0,
                                'is_pneumonia': pn_data.get('is_pneumonia', False),
                                'probabilities': pn_data.get('per_model', {}),
                                'severity': pn_data.get('severity', 'Unknown'),
                                'affected_area_pct': pn_data.get('affected_area_pct', 0),
                                'risk_level': pn_data.get('risk_level', 'Unknown'),
                                'risk_assessment': 'high' if pn_data.get('is_pneumonia') else 'low',
                                'heatmap_base64': pn_data.get('heatmap_base64'),
                                'confidence_tier': pn_data.get('confidence_tier', {})
                            }
                        else:
                            raise Exception(pn_data.get('error', 'Pneumonia API prediction failed'))
                    else:
                        raise Exception(f"Pneumonia API returned status {pn_response.status_code}")
                        
                except http_requests.exceptions.ConnectionError:
                    return jsonify({
                        'success': False,
                        'error': 'Pneumonia API is not available',
                        'message': 'Please start the Pneumonia API: python api/pneumonia_api.py --port 5003'
                    }), 503
                except http_requests.exceptions.Timeout:
                    return jsonify({
                        'success': False,
                        'error': 'Pneumonia API request timed out'
                    }), 504
                    

            
            processing_time = time.time() - start_time
            
            # Validate prediction result
            if not validate_prediction_data(prediction_result, image_type):
                raise ValueError("Invalid prediction result")
            
            # Get image info (safe for DICOM — PIL can't open .dcm)
            if filepath.lower().endswith('.dcm'):
                image_info = {
                    'format': 'DICOM',
                    'size': os.path.getsize(filepath),
                    'filename': os.path.basename(filepath)
                }
            else:
                image_info = get_image_info(filepath)
            
            # Prepare response
            response_data = {
                'success': True,
                'prediction': prediction_result,
                'image_type': image_type,
                'model_used': model_name,
                'image_info': {
                    'filename': filename,
                    'size': image_info.get('size', 'Unknown'),
                    'format': image_info.get('format', 'Unknown')
                },
                'processing_time': round(processing_time, 3),
                'timestamp': time.time(),
                'model_version': model_name
            }
            
            # Save to database if requested
            if save_result:
                try:
                    record_id = data_manager.save_prediction(
                        filepath, 
                        prediction_result, 
                        user_id=user_id,
                        processing_time=processing_time
                    )
                    response_data['record_id'] = record_id
                except Exception as e:
                    logger.warning(f"Failed to save prediction to database: {str(e)}")
            
            logger.info(f"Prediction completed: {prediction_result['predicted_class']} "
                       f"({prediction_result['confidence']:.3f}) in {processing_time:.3f}s")
            
            # Convert NumPy types to JSON-serializable types
            response_data = convert_numpy_types(response_data)
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['prediction_failed'],
                'details': str(e) if app.debug else None
            }), 500
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['unknown_error'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_tumor_base64():
    """Predict tumor from base64 encoded image."""
    global predictor
    
    try:
        # Check if model is loaded
        if predictor is None:
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['model_not_loaded']
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No base64 image data provided'
            }), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid base64 image data'
            }), 400
        
        # Get optional parameters
        user_id = data.get('user_id')
        enhance_image = data.get('enhance_image', False)
        
        logger.info("Processing base64 image")
        start_time = time.time()
        
        # Apply enhancement if requested
        if enhance_image:
            image_array = enhance_medical_image(image_array)
        
        # Make prediction
        prediction_result = predictor.predict(image_array)
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': prediction_result,
            'processing_time': round(processing_time, 3),
            'timestamp': time.time(),
            'model_version': 'ResNet50_v1'
        }
        
        logger.info(f"Base64 prediction completed: {prediction_result['predicted_class']} "
                   f"({prediction_result['confidence']:.3f}) in {processing_time:.3f}s")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Base64 prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['prediction_failed'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple images at once with routing based on imageType."""

    
    try:
        # Get imageType parameter to route to correct model
        image_type = request.form.get('imageType', 'tissue').lower()
        
        logger.info(f"📋 Batch predict request for imageType: {image_type}")
        
        # Route to appropriate model
        if image_type == 'tissue':
            # Brain tumor analysis uses Brain Tumor API
            model_name = "Brain Tumor Detection (EfficientNetB3)"
        elif image_type == 'pneumonia' or image_type == 'lung' or image_type == 'xray':
            # Pneumonia detection uses Pneumonia API
            image_type = 'pneumonia'
            model_name = "Pneumonia Detection (DenseNet121 + EfficientNet-B0 Ensemble)"
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid imageType: {image_type}. Must be "tissue" or "pneumonia"'
            }), 400
        
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image files provided'
            }), 400
        
        files = request.files.getlist('images')
        if len(files) > config.MAX_BATCH_SIZE:
            return jsonify({
                'success': False,
                'error': f'Too many files. Maximum batch size: {config.MAX_BATCH_SIZE}'
            }), 400
        
        user_id = request.form.get('user_id')
        enhance_images = request.form.get('enhance_images', 'false').lower() == 'true'
        
        results = []
        start_time = time.time()
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Invalid file'
                })
                continue
            
            try:
                # Route based on image type
                if image_type == 'tissue':
                    # For tissue/brain MRI: preprocess with PIL then proxy to Brain Tumor API
                    image = Image.open(file.stream)
                    image_array = np.array(image.convert('RGB'))
                    
                    if enhance_images:
                        image_array = enhance_medical_image(image_array)
                    
                    try:
                        import io
                        img_bytes = io.BytesIO()
                        Image.fromarray(image_array).save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        
                        files_to_send = {'image': (file.filename, img_bytes, 'image/png')}
                        bt_response = http_requests.post(
                            f"{BRAIN_TUMOR_API_URL}/analyze",
                            files=files_to_send,
                            timeout=60
                        )
                        
                        if bt_response.status_code == 200:
                            bt_data = bt_response.json()
                            if bt_data.get('success'):
                                prediction_result = {
                                    'predicted_class': bt_data['predicted_class'],
                                    'confidence': bt_data['confidence_percent'] / 100.0,
                                    'is_tumor': bt_data['predicted_class'] != 'notumor',
                                    'probabilities': bt_data.get('all_class_probabilities', {}),
                                    'risk_level': 'High Risk' if bt_data['predicted_class'] != 'notumor' else 'Low Risk',
                                    'risk_assessment': 'high' if bt_data['predicted_class'] != 'notumor' else 'low',
                                    # Grad-CAM fields — pass through from brain_tumor_api
                                    'heatmap_base64': bt_data.get('heatmap_base64'),
                                    'severity': bt_data.get('severity'),
                                    'affected_area_pct': bt_data.get('affected_area_pct'),
                                    'gradcam_enabled': bt_data.get('gradcam_enabled', False)
                                }
                                results.append({
                                    'filename': file.filename,
                                    'success': True,
                                    'prediction': prediction_result,
                                    'model_used': model_name
                                })
                            else:
                                results.append({
                                    'filename': file.filename,
                                    'success': False,
                                    'error': bt_data.get('error', 'Brain Tumor API prediction failed')
                                })
                        else:
                            results.append({
                                'filename': file.filename,
                                'success': False,
                                'error': f'Brain Tumor API returned status {bt_response.status_code}'
                            })
                    except http_requests.exceptions.ConnectionError:
                        logger.error('Brain Tumor API (port 5002) is not reachable')
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Brain Tumor API is not running on port 5002. Run: python api/brain_tumor_api.py --port 5002'
                        })
                    except http_requests.exceptions.Timeout:
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Brain Tumor API request timed out'
                        })
                    
                elif image_type == 'pneumonia':
                    # For pneumonia: send RAW file bytes directly to Pneumonia API
                    # (Pneumonia API has its own DICOM handler via pydicom — do NOT use PIL)
                    try:
                        import io
                        raw_bytes = file.stream.read()
                        img_bytes = io.BytesIO(raw_bytes)
                        is_dicom = file.filename.lower().endswith('.dcm')
                        if is_dicom:
                            mime_type = 'application/dicom'
                        elif file.filename.lower().endswith(('.jpg', '.jpeg')):
                            mime_type = 'image/jpeg'
                        else:
                            mime_type = 'image/png'
                        
                        files_to_send = {'image': (file.filename, img_bytes, mime_type)}
                        pn_response = http_requests.post(
                            f"{PNEUMONIA_API_URL}/analyze",
                            files=files_to_send,
                            timeout=60
                        )
                        
                        if pn_response.status_code == 200:
                            pn_data = pn_response.json()
                            if pn_data.get('success'):
                                prediction_result = {
                                    'predicted_class': pn_data['predicted_class'],
                                    'confidence': pn_data['confidence_percent'] / 100.0,
                                    'is_pneumonia': pn_data.get('is_pneumonia', False),
                                    'probabilities': pn_data.get('per_model', {}),
                                    'severity': pn_data.get('severity', 'Unknown'),
                                    'affected_area_pct': pn_data.get('affected_area_pct', 0),
                                    'risk_level': pn_data.get('risk_level', 'Unknown'),
                                    'risk_assessment': 'high' if pn_data.get('is_pneumonia') else 'low',
                                    'heatmap_base64': pn_data.get('heatmap_base64'),
                                    'confidence_tier': pn_data.get('confidence_tier', {})
                                }
                                results.append({
                                    'filename': file.filename,
                                    'success': True,
                                    'prediction': prediction_result,
                                    'model_used': model_name
                                })
                            else:
                                results.append({
                                    'filename': file.filename,
                                    'success': False,
                                    'error': pn_data.get('error', 'Pneumonia API prediction failed')
                                })
                        else:
                            results.append({
                                'filename': file.filename,
                                'success': False,
                                'error': f'Pneumonia API returned status {pn_response.status_code}'
                            })
                    except http_requests.exceptions.ConnectionError:
                        logger.error('Pneumonia API (port 5003) is not reachable')
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Pneumonia API is not running on port 5003. Run: python api/pneumonia_api.py --port 5003'
                        })
                    except http_requests.exceptions.Timeout:
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'Pneumonia API request timed out'
                        })
                    

                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'results': results,
            'image_type': image_type,
            'model_used': model_name,
            'total_images': len(files),
            'successful_predictions': sum(1 for r in results if r['success']),
            'total_processing_time': round(total_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['unknown_error'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """Get prediction history."""
    try:
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 100))
        
        predictions = data_manager.get_predictions(user_id=user_id, limit=limit)
        stats = data_manager.get_prediction_stats(user_id=user_id)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve prediction history'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics."""
    try:
        user_id = request.args.get('user_id')
        stats = data_manager.get_prediction_stats(user_id=user_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve statistics'
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information."""
    global predictor, pipeline
    
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['model_not_loaded']
            }), 500
        
        info = {
            'success': True,
            'model_type': 'ResNet50',
            'input_shape': config.MODEL_INPUT_SIZE,
            'classes': config.MODEL_CLASSES,
            'num_classes': config.NUM_CLASSES,
            'version': 'v1.0',
            'description': 'Pre-trained ResNet50 model fine-tuned for tumor detection',
            'pipeline_available': pipeline is not None
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve model information'
        }), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RecursiaDx ML API Gateway')
    parser.add_argument('--port', type=int, default=int(os.getenv('ML_API_PORT', 5001)),
                        help='Port to run the ML API gateway on (default: 5001)')
    args = parser.parse_args()

    logger.info("Starting RecursiaDx ML API Server")
    logger.info("=" * 70)
    
    # Initialize model
    if initialize_models():
        logger.info("✅ Server initialization successful")
        logger.info("=" * 70)
        
        # Run Flask app
        logger.info(f"🚀 Starting ML Gateway on port {args.port}...")
        app.run(
            host='0.0.0.0',
            port=args.port,
            debug=False,
            threaded=True
        )
    else:
        logger.error("❌ Server initialization failed")
        logger.error("Please check model files in ml/models/weights/")
        sys.exit(1)
