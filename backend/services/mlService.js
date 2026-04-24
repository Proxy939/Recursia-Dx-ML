import fetch from 'node-fetch';
import FormData from 'form-data';
import fs from 'fs';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5001';

class MLService {
  static async predictImage(imagePath, filename, imageType = 'tissue') {
    try {
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath), filename);
      formData.append('imageType', imageType);

      const response = await fetch(`${ML_API_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(60000) // 60 second timeout (pneumonia ensemble takes longer)
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'ML prediction failed');
      }

      return {
        success: true,
        prediction: result.prediction,
        metadata: result.metadata
      };
    } catch (error) {
      console.error('ML prediction error:', error);
      return {
        success: false,
        error: error.message,
        prediction: null
      };
    }
  }

  static async batchPredict(imagePaths, imageType = 'tissue') {
    try {
      const formData = new FormData();

      // Add imageType to form data for ML API routing
      formData.append('imageType', imageType);

      imagePaths.forEach(({ path, filename }) => {
        const isDicom = filename.toLowerCase().endsWith('.dcm');
        const contentType = isDicom ? 'application/dicom' : 'application/octet-stream';
        formData.append('images', fs.createReadStream(path), {
          filename: filename,
          contentType: contentType
        });
      });

      const response = await fetch(`${ML_API_URL}/batch_predict`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(120000) // 120 second timeout for batch
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Batch ML prediction failed');
      }

      return {
        success: true,
        predictions: result.results, // ML server returns 'results' not 'predictions'
        imageType: result.image_type, // Return imageType from ML response
        modelUsed: result.model_used, // Return which model was used
        summary: {
          total_images: result.total_images,
          successful_predictions: result.successful_predictions,
          total_processing_time: result.total_processing_time
        },
        metadata: result.metadata || {}
      };
    } catch (error) {
      console.error('Batch ML prediction error:', error);
      return {
        success: false,
        error: error.message,
        predictions: []
      };
    }
  }

  static async checkHealth(retries = 6, delayMs = 15000) {
    // Retry up to `retries` times with `delayMs` gap.
    // ML models can take 30-90s to load on first startup.
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await fetch(`${ML_API_URL}/health`, {
          signal: AbortSignal.timeout(10000) // 10s per attempt
        });

        if (response.ok) {
          const result = await response.json();
          console.log(`✅ ML Gateway healthy on attempt ${attempt}`);
          return { healthy: true, details: result };
        }

        console.warn(`⚠️  ML Gateway returned ${response.status} on attempt ${attempt}/${retries}`);
      } catch (error) {
        console.warn(`⚠️  ML Gateway not ready (attempt ${attempt}/${retries}): ${error.message}`);
      }

      if (attempt < retries) {
        console.log(`   Waiting ${delayMs / 1000}s before retry...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    return {
      healthy: false,
      error: `ML Gateway on port 5001 did not respond after ${retries} attempts. Make sure it is running: python api/app.py --port 5001`
    };
  }

  static async getModelInfo() {
    try {
      const response = await fetch(`${ML_API_URL}/model_info`, {
        timeout: 5000
      });

      if (!response.ok) {
        throw new Error(`Model info failed: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: true,
        modelInfo: result
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  static async generateHeatmap(imagePath, filename) {
    try {
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath), filename);

      const response = await fetch(`${ML_API_URL}/generate_heatmap`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(120000) // 2 minute timeout for heatmap generation
      });

      if (!response.ok) {
        throw new Error(`Heatmap generation failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Heatmap generation failed');
      }

      return {
        success: true,
        heatmap: result.heatmap,
        prediction: result.prediction,
        processingTime: result.processing_time
      };
    } catch (error) {
      console.error('Heatmap generation error:', error);
      return {
        success: false,
        error: error.message,
        heatmap: null
      };
    }
  }
}

export default MLService;