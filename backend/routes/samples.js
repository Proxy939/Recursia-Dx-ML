import express from 'express';
import Sample from '../models/Sample.js';
import { verifyToken, authorize, authorizeOwnerOrAdmin } from '../middleware/auth.js';
import { sampleValidations, queryValidations } from '../middleware/validation.js';
import { catchAsync, AppError } from '../middleware/errorHandler.js';
import upload from '../middleware/upload.js';
import MLService from '../services/mlService.js';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import fs from 'fs';

// ESM __dirname equivalent
const __filename_routes = fileURLToPath(import.meta.url);
const __dirname_routes = path.dirname(__filename_routes); // = backend/routes
const PROJECT_ROOT = path.resolve(__dirname_routes, '..', '..'); // = project root

const router = express.Router();

// Generate real heatmap function using ML service
async function generateRealHeatmap(imagePath, imageId) {
  return new Promise(async (resolve) => {
    try {
      const FormData = (await import('form-data')).default;
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath));

      // Call ML service to generate real heatmap
      const response = await fetch('http://localhost:5001/generate_heatmap', {
        method: 'POST',
        body: formData,
        timeout: 30000
      });

      if (!response.ok) {
        throw new Error(`ML heatmap service returned ${response.status}`);
      }

      const result = await response.json();

      if (result.success && result.heatmap) {
        const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

        // Ensure output directory exists
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }

        const timestamp = Date.now();
        const outputFilename = `real_heatmap_${imageId}_${timestamp}.png`;
        const outputPath = path.join(outputDir, outputFilename);

        // Save base64 heatmap to file
        if (result.heatmap.base64) {
          const base64Data = result.heatmap.base64.replace(/^data:image\/png;base64,/, '');
          fs.writeFileSync(outputPath, base64Data, 'base64');

          console.log(`✅ Real heatmap generated: ${outputFilename}`);

          resolve({
            success: true,
            heatmap: {
              filename: outputFilename,
              path: `/api/samples/heatmap/${outputFilename}`,
              base64: result.heatmap.base64,
              type: result.heatmap.type || 'tumor_probability',
              colormap: result.heatmap.colormap || 'jet',
              analytics: result.heatmap.analytics || {}
            }
          });
        } else {
          throw new Error('No heatmap data received from ML service');
        }
      } else {
        throw new Error(result.error || 'Heatmap generation failed');
      }

    } catch (error) {
      console.error(`❌ Real heatmap generation error for image ${imageId}:`, error.message);
      resolve({ success: false, error: error.message });
    }
  });
}

// REMOVED: Old random generateAutoHeatmap function - now using real ML heatmaps only

// Simple test route without any middleware
router.post('/test-upload', (req, res) => {
  console.log('🎯 TEST ROUTE REACHED - No Auth Required!');
  res.json({ success: true, message: 'Test upload works without auth!' });
});

// Test ML service health status (no auth required for debugging)
router.get('/ml-health-test', catchAsync(async (req, res) => {
  try {
    const response = await fetch(`${process.env.ML_API_URL || 'http://localhost:5000'}/health`, {
      method: 'GET',
      timeout: 5000
    });

    if (!response.ok) {
      throw new Error(`ML API returned ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();

    res.json({
      success: true,
      data: {
        mlService: {
          status: 'healthy',
          details: result,
          url: 'http://localhost:5001'
        }
      }
    });

  } catch (error) {
    console.error('ML Health check failed:', error.message);

    res.json({
      success: false,
      data: {
        mlService: {
          status: 'unavailable',
          error: error.message,
          url: 'http://localhost:5001',
          suggestion: 'Please ensure the ML API server is running on port 5001'
        }
      }
    });
  }
}));

// Serve uploaded images
router.get('/image/:filename', (req, res) => {
  const filename = req.params.filename;
  const imagePath = path.join(process.cwd(), 'uploads', filename);

  // Check if file exists
  if (!require('fs').existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image not found' });
  }

  res.sendFile(imagePath);
});

// Generate heatmap for an image (no auth required for testing)
router.post('/generate-heatmap', upload.single('image'), catchAsync(async (req, res) => {
  console.log('🎨 Heatmap generation endpoint called');

  try {
    const { heatmapType = 'tumor_probability', colormap = 'hot' } = req.body;
    let imagePath;

    // Handle image upload or existing image
    if (req.file) {
      imagePath = req.file.path;
      console.log('📷 Using uploaded image:', imagePath);
    } else if (req.body.imagePath) {
      imagePath = req.body.imagePath;
      console.log('📁 Using existing image:', imagePath);
    } else {
      return res.status(400).json({
        success: false,
        error: 'No image provided. Upload an image or provide imagePath.'
      });
    }

    // Check if image exists
    if (!fs.existsSync(imagePath)) {
      return res.status(404).json({
        success: false,
        error: 'Image file not found'
      });
    }

    console.log('🔬 Generating heatmap...');
    console.log(`   Type: ${heatmapType}`);
    console.log(`   Colormap: ${colormap}`);
    console.log(`   Image: ${imagePath}`);

    // Call Python heatmap generation script
    const pythonScript = path.join(process.cwd(), '..', 'ml', 'web_heatmap_generator.py');
    const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputFilename = `heatmap_${Date.now()}_${heatmapType}_${colormap}.png`;
    const outputPath = path.join(outputDir, outputFilename);

    // Execute Python script
    const pythonProcess = spawn('python', [
      pythonScript,
      '--image', imagePath,
      '--output', outputPath,
      '--type', heatmapType,
      '--colormap', colormap,
      '--format', 'web'
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Heatmap generated successfully');

        // Check if output file was created
        if (fs.existsSync(outputPath)) {
          // Convert to base64 for web display
          const imageBuffer = fs.readFileSync(outputPath);
          const base64Image = imageBuffer.toString('base64');

          // Parse any JSON output from Python script
          let analytics = {};
          try {
            const jsonMatch = stdout.match(/ANALYTICS_JSON:(.*?)END_ANALYTICS/s);
            if (jsonMatch) {
              analytics = JSON.parse(jsonMatch[1]);
            }
          } catch (e) {
            console.log('No analytics data found in output');
          }

          res.json({
            success: true,
            data: {
              heatmap: {
                type: heatmapType,
                colormap: colormap,
                image_base64: `data:image/png;base64,${base64Image}`,
                file_path: `/api/samples/heatmap/${outputFilename}`,
                analytics: analytics
              },
              original_image: imagePath,
              processing_time: Date.now() - parseInt(outputFilename.split('_')[1])
            }
          });
        } else {
          console.error('❌ Output file not created');
          res.status(500).json({
            success: false,
            error: 'Heatmap file was not created',
            debug: { stdout, stderr }
          });
        }
      } else {
        console.error('❌ Python script failed with code:', code);
        console.error('stderr:', stderr);

        res.status(500).json({
          success: false,
          error: 'Heatmap generation failed',
          details: stderr,
          debug: { stdout, stderr, code }
        });
      }
    });

  } catch (error) {
    console.error('❌ Heatmap generation error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error during heatmap generation',
      details: error.message
    });
  }
}));

// Serve generated heatmaps
router.get('/heatmap/:filename', (req, res) => {
  const filename = req.params.filename;
  const heatmapPath = path.join(process.cwd(), 'uploads', 'heatmaps', filename);

  if (!fs.existsSync(heatmapPath)) {
    return res.status(404).json({ error: 'Heatmap not found' });
  }

  res.sendFile(heatmapPath);
});

// Test endpoint to get demo sample without auth (for testing heatmaps)
router.get('/demo-sample', catchAsync(async (req, res) => {
  try {
    const demoSample = await Sample.findOne({ sampleId: 'SP-2025-DEMO-001' });

    if (!demoSample) {
      return res.status(404).json({
        success: false,
        message: 'Demo sample not found'
      });
    }

    res.json({
      success: true,
      data: {
        sample: demoSample
      }
    });
  } catch (error) {
    console.error('Error fetching demo sample:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch demo sample',
      error: error.message
    });
  }
}));

// Upload sample with images and ML analysis (auth temporarily disabled for testing)
router.post('/upload-with-analysis',
  (req, res, next) => {
    // Wrap multer to catch file filter errors gracefully
    upload.array('images', 10)(req, res, (err) => {
      if (err) {
        console.error('📁 Multer upload error:', err.message);
        // Handle multer-specific errors
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({
            success: false,
            message: 'File too large. Maximum size is 50MB per file.'
          });
        }
        if (err.code === 'LIMIT_FILE_COUNT' || err.code === 'LIMIT_UNEXPECTED_FILE') {
          return res.status(400).json({
            success: false,
            message: 'Too many files. Maximum is 10 files per upload.'
          });
        }
        // File filter rejection (invalid file type)
        return res.status(400).json({
          success: false,
          message: err.message || 'Invalid file type. Supported: JPEG, PNG, BMP, DICOM (.dcm)'
        });
      }
      next();
    });
  },
  catchAsync(async (req, res) => {
    console.log('🚀 UPLOAD ROUTE REACHED - Auth Disabled!');
    try {
      // Parse sample data or patient info
      let sampleData
      if (req.body.sampleData) {
        sampleData = JSON.parse(req.body.sampleData)
        console.log('📦 Parsed sampleData:', JSON.stringify(sampleData, null, 2));
      } else if (req.body.patientInfo) {
        // If only patientInfo is sent, create minimal sample data
        const patientInfo = JSON.parse(req.body.patientInfo)
        sampleData = {
          patientInfo,
          sampleType: 'Brain MRI',
          collectionDate: new Date().toISOString()
        }
      } else {
        // No data provided, use defaults
        sampleData = {
          patientInfo: {
            name: 'Unknown Patient',
            age: 0,
            gender: 'Other'
          },
          sampleType: 'Brain MRI',
          collectionDate: new Date().toISOString()
        }
      }

      // Extract imageType from sampleData (frontend sends it nested in patientData)
      const imageType = sampleData.imageType || req.body.imageType || 'tissue';
      console.log(`📋 Processing ${imageType} image(s)`);
      console.log(`🔍 DEBUG: sampleData.imageType=${sampleData.imageType}, req.body.imageType=${req.body.imageType}, final imageType=${imageType}`);

      // Set sampleType based on imageType selection
      sampleData.sampleType = (imageType === 'pneumonia' || imageType === 'lung' || imageType === 'xray') ? 'Chest X-ray' :
                               'Brain MRI';

      // CRITICAL: Check if ML service is available
      const mlHealthCheck = await MLService.checkHealth();
      if (!mlHealthCheck.healthy) {
        return res.status(503).json({
          success: false,
          message: 'ML service is not available. Please start the ML server on port 5001.',
          error: 'ML_SERVICE_UNAVAILABLE'
        });
      }

      // Generate unique sample ID
      const sampleCount = await Sample.countDocuments();
      const sampleId = `SP-${new Date().getFullYear()}-${String(sampleCount + 1).padStart(4, '0')}`;

      // Process uploaded images
      const processedImages = [];

      if (req.files && req.files.length > 0) {
        console.log(`Processing ${req.files.length} uploaded ${imageType} images...`);

        // Prepare images for ML analysis with imageType
        const imagePathsForML = req.files.map(file => ({
          path: file.path,
          filename: file.filename
        }));

        // Run batch ML analysis - REAL ANALYSIS ONLY, passing imageType
        console.log(`🧠 Running real ML analysis for ${imageType} images...`);
        const mlResults = await MLService.batchPredict(imagePathsForML, imageType);

        console.log(`📊 ML Results received:`, JSON.stringify(mlResults, null, 2));

        if (!mlResults.success) {
          // Gateway itself failed (network error to ML gateway on 5001)
          return res.status(503).json({
            success: false,
            message: `ML Gateway unreachable on port 5001. Make sure it is running: python api/app.py --port 5001`,
            error: mlResults.error
          });
        }

        // Check if ALL individual predictions failed (e.g., downstream API is down)
        const allFailed = mlResults.predictions && mlResults.predictions.length > 0 &&
                          mlResults.predictions.every(p => !p.success);
        if (allFailed) {
          const firstError = mlResults.predictions[0]?.error || 'Unknown error';
          return res.status(503).json({
            success: false,
            message: firstError,
            error: 'ALL_PREDICTIONS_FAILED'
          });
        }


        // Process each image with its ML result
        for (let i = 0; i < req.files.length; i++) {
          const file = req.files[i];
          const mlResult = mlResults.success ? mlResults.predictions[i] : null;
          
          // Convert DICOM files to PNG for browser display
          let displayFilename = file.filename;
          let displayUrl = `/uploads/${file.filename}`;
          const isDicom = file.originalname.toLowerCase().endsWith('.dcm');
          
          if (isDicom) {
            try {
              const { execSync } = await import('child_process');
              const pngFilename = file.filename.replace(/\.dcm$/i, '.png');
              const uploadsDir = path.resolve(__dirname_routes, '..', 'uploads');
              const pngPath = path.join(uploadsDir, pngFilename);
              const dcm2pngScript = path.join(PROJECT_ROOT, 'ml', 'utils', 'dcm2png.py');
              
              console.log(`📸 Converting DICOM: ${file.filename}`);
              console.log(`   Script: ${dcm2pngScript}`);
              console.log(`   Input:  ${file.path}`);
              console.log(`   Output: ${pngPath}`);
              
              if (!fs.existsSync(dcm2pngScript)) {
                throw new Error(`dcm2png.py not found at ${dcm2pngScript}`);
              }
              
              // Convert using standalone Python script (no ML proxy dependency)
              execSync(`python "${dcm2pngScript}" "${file.path}" "${pngPath}"`, {
                timeout: 30000,
                stdio: 'pipe'
              });
              
              displayFilename = pngFilename;
              displayUrl = `/uploads/${pngFilename}`;
              console.log(`✅ DICOM → PNG success: ${pngFilename}`);
            } catch (convertErr) {
              console.error(`❌ DICOM conversion FAILED: ${convertErr.message}`);
              // Fallback: the server.js middleware will convert on-the-fly when requested
            }
          }

          const imageData = {
            filename: displayFilename,
            originalName: file.originalname,
            mimetype: isDicom ? 'image/png' : file.mimetype,
            size: file.size,
            path: file.path,
            url: displayUrl, // Points to displayable PNG (or original if not DICOM)
            isDicom: isDicom,
            originalDicomPath: isDicom ? file.path : undefined,
            uploadedBy: null, // Temporarily removed for testing
            uploadedAt: new Date()
          };

          // Add ML analysis if available
          if (mlResult && mlResult.prediction) {
            const prediction = mlResult.prediction;

            // Handle different result formats based on imageType
            if (imageType === 'pneumonia' || imageType === 'lung' || imageType === 'xray') {
              // Pneumonia analysis - chest X-ray
              imageData.mlAnalysis = {
                prediction: prediction.predicted_class || (prediction.is_pneumonia ? 'Pneumonia' : 'Normal'),
                confidence: Number(prediction.confidence) || 0.5,
                riskAssessment: prediction.is_pneumonia ? 'high' : 'low',
                processingTime: prediction.processing_time || 0,
                imageId: prediction.image_id || file.filename,
                modelVersion: 'DenseNet121+EfficientNetB0-v1.0',
                analyzedAt: new Date(),
                isPneumonia: prediction.is_pneumonia || false,
                severity: prediction.severity || 'Unknown',
                affectedAreaPct: prediction.affected_area_pct || 0,
                confidenceTier: prediction.confidence_tier || {},
                classProbabilities: prediction.probabilities || {},
                metadata: mlResult
              };
            } else {
              // Tissue analysis - tumor detection (4-class: glioma, meningioma, pituitary, notumor)
              const mapPrediction = (predicted_class) => {
                if (!predicted_class) return 'indeterminate';
                const cls = predicted_class.toLowerCase();
                if (cls === 'notumor' || cls === 'no tumor' || cls === 'non-tumor') return 'benign';
                if (['glioma', 'meningioma', 'pituitary', 'tumor'].includes(cls)) return 'malignant';
                return 'indeterminate';
              };

              // Map tumor type for 4-class classification display
              const mapTumorType = (predicted_class) => {
                if (!predicted_class) return null;
                const cls = predicted_class.toLowerCase();
                const tumorTypes = {
                  'glioma': { name: 'Glioma', description: 'A tumor arising from glial cells in the brain or spine', severity: 'High' },
                  'meningioma': { name: 'Meningioma', description: 'A tumor arising from the meninges, the membranes surrounding the brain and spinal cord', severity: 'Moderate' },
                  'pituitary': { name: 'Pituitary Tumor', description: 'An abnormal growth in the pituitary gland at the base of the brain', severity: 'Moderate' },
                  'notumor': { name: 'No Tumor', description: 'No tumor detected in the brain MRI scan', severity: 'None' }
                };
                return tumorTypes[cls] || null;
              };

              const mapRiskAssessment = (risk) => {
                if (risk && typeof risk === 'string') {
                  switch (risk.toLowerCase()) {
                    case 'low risk': return 'low';
                    case 'medium risk': return 'medium';
                    case 'high risk': return 'high';
                    default: return 'medium';
                  }
                }
                return 'medium';
              };

              imageData.mlAnalysis = {
                prediction: mapPrediction(prediction.predicted_class),
                confidence: Number(prediction.confidence) || 0.5,
                riskAssessment: mapRiskAssessment(prediction.risk_assessment),
                processingTime: prediction.processing_time || 0,
                imageId: prediction.image_id || file.filename,
                modelVersion: 'EfficientNetB3-v1.0',
                analyzedAt: new Date(),
                tumorType: prediction.predicted_class || null,
                tumorTypeInfo: mapTumorType(prediction.predicted_class),
                classProbabilities: prediction.probabilities || {},
                metadata: mlResult
              };
            }
          }

          // Generate REAL heatmap using ML service - Only for tissue images
          if (imageType === 'tissue') {
            console.log(`🎨 Generating real heatmap for image ${i + 1}/${req.files.length}`);
            const heatmapResult = await generateRealHeatmap(file.path, file.filename);

            if (heatmapResult.success) {
              imageData.heatmap = heatmapResult.heatmap;
              console.log(`✅ Real heatmap generated for ${file.filename}`);
            } else {
              console.log(`⚠️ Heatmap generation skipped for ${file.filename}: ${heatmapResult.error}`);
              // Don't add mock heatmap - leave it undefined
            }
          } else if (imageType === 'pneumonia' || imageType === 'lung' || imageType === 'xray') {
            // Pneumonia analysis returns its own heatmap from the API
            if (mlResult && mlResult.prediction && mlResult.prediction.heatmap_base64) {
              imageData.heatmap = {
                base64: `data:image/png;base64,${mlResult.prediction.heatmap_base64}`,
                type: 'gradcam',
                colormap: 'jet'
              };
              console.log(`✅ Pneumonia Grad-CAM heatmap stored for ${file.filename}`);
            }
          }

          processedImages.push(imageData);
        }

        // Calculate overall AI analysis from processed image data
        if (processedImages.length > 0 && processedImages.some(img => img.mlAnalysis)) {
          const mlAnalyses = processedImages.filter(img => img.mlAnalysis).map(img => img.mlAnalysis);
          const malignantCount = mlAnalyses.filter(ml => ml.prediction === 'malignant').length;
          const highRiskCount = mlAnalyses.filter(ml => ml.riskAssessment === 'high').length;
          const avgConfidence = mlAnalyses.reduce((sum, ml) => sum + (ml.confidence || 0), 0) / mlAnalyses.length;

          sampleData.aiAnalysis = {
            overallPrediction: malignantCount > mlAnalyses.length / 2 ? 'malignant' : 'benign',
            averageConfidence: Number(avgConfidence.toFixed(3)) || 0.5,
            highRiskImages: highRiskCount,
            totalImagesAnalyzed: mlAnalyses.length,
            recommendations: malignantCount > 0 ? ['Recommend pathologist review', 'Consider additional testing'] : ['Routine monitoring'],
            flaggedFindings: highRiskCount > 0 ? ['High-risk features detected'] : [],
            batchAnalyzedAt: new Date(),
            modelInfo: {
              name: 'ResNet50-TumorClassifier',
              version: '1.0.0',
              accuracy: 0.94
            }
          };
        }
      }

      // Ensure required fields have valid values before saving
      if (!sampleData.patientInfo) sampleData.patientInfo = {};
      sampleData.patientInfo.patientId = sampleData.patientInfo.patientId || `PT-${Date.now()}`;
      sampleData.patientInfo.name = sampleData.patientInfo.name || 'Anonymous Patient';
      sampleData.patientInfo.age = parseInt(sampleData.patientInfo.age) || 0;
      // Ensure gender is a valid enum value
      const validGenders = ['Male', 'Female', 'Other'];
      if (!validGenders.includes(sampleData.patientInfo.gender)) {
        sampleData.patientInfo.gender = 'Other';
      }
      // Ensure collectionInfo has a date
      if (!sampleData.collectionInfo) sampleData.collectionInfo = {};
      sampleData.collectionInfo.collectionDate = sampleData.collectionInfo.collectionDate || new Date();

      // Create sample with all data
      const sample = await Sample.create({
        sampleId,
        ...sampleData,
        images: processedImages,
        submittedBy: null, // Temporarily removed for testing
        workflow: {
          receivedAt: new Date(),
          estimatedCompletionTime: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        }
      });

      // Skip population for test route since no user authentication
      // await sample.populate([
      //   { path: 'submittedBy', select: 'name email role' },
      //   { path: 'images.uploadedBy', select: 'name' }
      // ]);

      res.status(201).json({
        success: true,
        message: 'Sample uploaded and analyzed successfully',
        data: {
          sample,
          mlAnalysis: req.files ? {
            imagesAnalyzed: req.files.length,
            aiInsights: sample.aiAnalysis
          } : null
        }
      });
    } catch (error) {
      console.error('❌ Upload with analysis error:', error);
      res.status(500).json({
        success: false,
        message: 'Upload failed',
        error: error.message
      });
    }
  })
);

// Create new sample
router.post('/',
  authorize('Pathologist', 'Lab Technician', 'Admin'),
  sampleValidations.create,
  catchAsync(async (req, res) => {
    const sampleData = {
      ...req.body,
      submittedBy: req.user._id
    };

    const sample = await Sample.create(sampleData);
    await sample.populate('submittedBy', 'name email role');

    res.status(201).json({
      success: true,
      message: 'Sample created successfully',
      data: {
        sample
      }
    });
  })
);

// Get all samples with filtering and pagination
router.get('/',
  queryValidations.pagination,
  queryValidations.dateRange,
  catchAsync(async (req, res) => {
    const {
      page = 1,
      limit = 20,
      status,
      specimenType,
      priority,
      submittedBy,
      startDate,
      endDate,
      search
    } = req.query;

    // Build filter object
    const filter = {};

    // Role-based filtering
    if (req.user.role === 'Lab Technician') {
      filter.submittedBy = req.user._id;
    }

    if (status) filter.status = status;
    if (specimenType) filter.specimenType = specimenType;
    if (priority) filter.priority = priority;
    if (submittedBy && ['Admin', 'Pathologist'].includes(req.user.role)) {
      filter.submittedBy = submittedBy;
    }

    // Date range filter
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }

    // Search filter
    if (search) {
      filter.$or = [
        { sampleId: { $regex: search, $options: 'i' } },
        { 'patientInfo.name': { $regex: search, $options: 'i' } },
        { 'patientInfo.patientId': { $regex: search, $options: 'i' } },
        { anatomicalSite: { $regex: search, $options: 'i' } }
      ];
    }

    // Calculate pagination
    const skip = (page - 1) * limit;

    // Execute query
    const [samples, totalCount] = await Promise.all([
      Sample.find(filter)
        .populate('submittedBy', 'name email role')
        .populate('assignedTo', 'name email role')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(parseInt(limit)),
      Sample.countDocuments(filter)
    ]);

    const totalPages = Math.ceil(totalCount / limit);

    res.json({
      success: true,
      data: {
        samples,
        pagination: {
          currentPage: parseInt(page),
          totalPages,
          totalCount,
          hasNext: page < totalPages,
          hasPrev: page > 1
        }
      }
    });
  })
);

// Get sample by ID
router.get('/:id',
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id)
      .populate('submittedBy', 'name email role department')
      .populate('assignedTo', 'name email role department');

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Check authorization
    if (req.user.role === 'Lab Technician' &&
      sample.submittedBy._id.toString() !== req.user._id.toString()) {
      return next(new AppError('Not authorized to view this sample', 403));
    }

    res.json({
      success: true,
      data: {
        sample
      }
    });
  })
);

// Update sample
router.put('/:id',
  sampleValidations.update,
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Check authorization
    const canEdit = req.user.role === 'Admin' ||
      req.user.role === 'Pathologist' ||
      (req.user.role === 'Lab Technician' &&
        sample.submittedBy.toString() === req.user._id.toString());

    if (!canEdit) {
      return next(new AppError('Not authorized to update this sample', 403));
    }

    // Prevent certain status changes based on role
    if (req.body.status) {
      const statusTransitions = {
        'Lab Technician': ['Received', 'Processing', 'Sectioning'],
        'Pathologist': ['Reading', 'Reporting', 'Complete'],
        'Admin': ['Received', 'Processing', 'Sectioning', 'Staining', 'Reading', 'Reporting', 'Complete', 'Cancelled']
      };

      const allowedStatuses = statusTransitions[req.user.role] || [];
      if (!allowedStatuses.includes(req.body.status)) {
        return next(new AppError(`Role ${req.user.role} cannot set status to ${req.body.status}`, 403));
      }
    }

    const updatedSample = await Sample.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    ).populate('submittedBy assignedTo', 'name email role');

    res.json({
      success: true,
      message: 'Sample updated successfully',
      data: {
        sample: updatedSample
      }
    });
  })
);

// Delete sample (soft delete)
router.delete('/:id',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Prevent deletion if sample has been processed
    if (['Reading', 'Reporting', 'Complete'].includes(sample.status)) {
      return next(new AppError('Cannot delete sample that has been processed', 400));
    }

    await Sample.findByIdAndUpdate(req.params.id, {
      status: 'Cancelled',
      cancelledAt: new Date(),
      cancelledBy: req.user._id
    });

    res.json({
      success: true,
      message: 'Sample deleted successfully'
    });
  })
);

// Assign sample to pathologist
router.put('/:id/assign',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res, next) => {
    const { assignedTo } = req.body;

    if (!assignedTo) {
      return next(new AppError('Assigned user ID is required', 400));
    }

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    const updatedSample = await Sample.findByIdAndUpdate(
      req.params.id,
      {
        assignedTo,
        assignedAt: new Date(),
        status: sample.status === 'Received' ? 'Processing' : sample.status
      },
      { new: true, runValidators: true }
    ).populate('submittedBy assignedTo', 'name email role');

    res.json({
      success: true,
      message: 'Sample assigned successfully',
      data: {
        sample: updatedSample
      }
    });
  })
);

// Add image to sample
router.post('/:id/images',
  catchAsync(async (req, res, next) => {
    const { fileName, filePath, fileSize, magnification, staining } = req.body;

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    const imageData = {
      fileName,
      filePath,
      fileSize,
      uploadedBy: req.user._id,
      uploadedAt: new Date(),
      metadata: {
        magnification,
        staining
      }
    };

    sample.images.push(imageData);
    await sample.save();

    res.json({
      success: true,
      message: 'Image added successfully',
      data: {
        image: imageData
      }
    });
  })
);

// Update sample status with workflow tracking
router.put('/:id/status',
  catchAsync(async (req, res, next) => {
    const { status, notes } = req.body;

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Add workflow entry
    const workflowEntry = {
      status,
      timestamp: new Date(),
      user: req.user._id,
      notes
    };

    sample.workflow.push(workflowEntry);
    sample.status = status;

    // Update specific timestamps based on status
    const timestampMap = {
      'Received': 'receivedAt',
      'Processing': 'processingStartedAt',
      'Complete': 'completedAt'
    };

    if (timestampMap[status]) {
      sample[timestampMap[status]] = new Date();
    }

    await sample.save();

    res.json({
      success: true,
      message: 'Status updated successfully',
      data: {
        sample
      }
    });
  })
);

// Get sample statistics
router.get('/stats/overview',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res) => {
    const { startDate, endDate } = req.query;

    const matchStage = {};
    if (startDate || endDate) {
      matchStage.createdAt = {};
      if (startDate) matchStage.createdAt.$gte = new Date(startDate);
      if (endDate) matchStage.createdAt.$lte = new Date(endDate);
    }

    const stats = await Sample.aggregate([
      { $match: matchStage },
      {
        $group: {
          _id: {
            status: '$status',
            specimenType: '$specimenType',
            priority: '$priority'
          },
          count: { $sum: 1 },
          avgProcessingTime: { $avg: '$processingTime' }
        }
      },
      {
        $group: {
          _id: null,
          totalSamples: { $sum: '$count' },
          statusBreakdown: {
            $push: {
              status: '$_id.status',
              count: '$count'
            }
          },
          specimenTypeBreakdown: {
            $push: {
              type: '$_id.specimenType',
              count: '$count'
            }
          },
          priorityBreakdown: {
            $push: {
              priority: '$_id.priority',
              count: '$count'
            }
          },
          avgProcessingTime: { $avg: '$avgProcessingTime' }
        }
      }
    ]);

    res.json({
      success: true,
      data: {
        stats: stats[0] || {}
      }
    });
  })
);

// List available heatmaps
router.get('/heatmaps/list', catchAsync(async (req, res) => {
  try {
    const heatmapDir = path.join(process.cwd(), 'uploads', 'heatmaps');

    if (!fs.existsSync(heatmapDir)) {
      return res.json({ success: true, heatmaps: [] });
    }

    const files = fs.readdirSync(heatmapDir);
    const heatmaps = files
      .filter(file => file.endsWith('.png') || file.endsWith('.jpg'))
      .slice(0, 10) // Limit to 10 most recent
      .map((file, index) => ({
        id: `heatmap-${index}`,
        name: file.replace(/^auto_heatmap_/, '').replace(/\.[^/.]+$/, ''),
        src: `/uploads/heatmaps/${file}`,
        // TODO: Extract real metadata from heatmap generation or database
        confidence: null, // Real confidence should come from ML analysis
        prediction: null  // Real prediction should come from ML analysis
      }));

    res.json({
      success: true,
      heatmaps
    });
  } catch (error) {
    console.error('Error listing heatmaps:', error);
    res.json({ success: true, heatmaps: [] });
  }
}));

// ================================================================
// Demo Analysis Endpoint - REAL ML Predictions
// ================================================================
router.post('/demo-analysis', catchAsync(async (req, res) => {
  const demoType = req.query.type || 'tumor';
  console.log(`🧪 Demo analysis requested: ${demoType}`);

  const isDemoTumor = demoType === 'tumor';

  // Demo images from demo_images folder
  const demoImagesDir = path.join(process.cwd(), '..', 'demo_images');
  const demoImageFilename = isDemoTumor ? 'cancer.jpg' : 'non_tumor_sample.jpg';
  const demoImagePath = path.join(demoImagesDir, demoImageFilename);

  console.log(`📁 Demo image path: ${demoImagePath}`);

  // Check if demo image exists
  if (!fs.existsSync(demoImagePath)) {
    console.error(`❌ Demo image not found: ${demoImagePath}`);
    return res.status(404).json({
      success: false,
      error: `Demo image not found. Please add ${demoImageFilename} to demo_images folder.`
    });
  }

  // Copy image to uploads folder for frontend access
  const uploadsDir = path.join(process.cwd(), 'uploads');
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  const copiedFilename = `demo_${Date.now()}_${demoImageFilename}`;
  const copiedPath = path.join(uploadsDir, copiedFilename);
  fs.copyFileSync(demoImagePath, copiedPath);
  console.log(`📋 Demo image copied to: ${copiedPath}`);

  const imageUrl = `/uploads/${copiedFilename}`;

  // ================================================================
  // REAL ML PREDICTION via Brain Tumor API
  // ================================================================
  let mlResult = null;

  try {
    console.log('🧠 Sending image to Brain Tumor ML service for REAL prediction...');

    // Use MLService to get real predictions
    const mlResults = await MLService.batchPredict(
      [{ path: copiedPath, filename: copiedFilename }],
      'tissue'
    );

    console.log('📊 ML Results:', JSON.stringify(mlResults, null, 2));

    if (mlResults.success && mlResults.predictions && mlResults.predictions.length > 0) {
      const prediction = mlResults.predictions[0];
      if (prediction.success && prediction.prediction) {
        mlResult = {
          predicted_class: prediction.prediction.predicted_class,
          confidence: prediction.prediction.confidence,
          is_tumor: prediction.prediction.is_tumor,
          probabilities: prediction.prediction.probabilities || {},
          risk_level: prediction.prediction.risk_level,
          risk_assessment: prediction.prediction.risk_assessment || 'medium'
        };
        console.log(`✅ Real ML prediction: ${mlResult.predicted_class} @ ${(mlResult.confidence * 100).toFixed(1)}%`);
      }
    }
  } catch (mlError) {
    console.error('⚠️ ML service error:', mlError.message);
  }

  // If ML failed, return error (no fallback to fake data)
  if (!mlResult) {
    return res.status(503).json({
      success: false,
      error: 'ML service unavailable. Please ensure Brain Tumor API is running on port 5002.',
      message: 'Start Brain Tumor API with: cd ml/api && python brain_tumor_api.py --port 5002'
    });
  }

  // Generate unique sample ID for demo
  const sampleCount = await Sample.countDocuments();
  const sampleId = `DEMO-${new Date().getFullYear()}-${String(sampleCount + 1).padStart(4, '0')}`;

  // Map prediction to backend format
  const mapPrediction = (predicted_class) => {
    switch (predicted_class) {
      case 'Non-Tumor': return 'benign';
      case 'Tumor': return 'malignant';
      default: return 'indeterminate';
    }
  };

  // Create demo sample in database with REAL ML results
  const demoSample = new Sample({
    sampleId,
    patientInfo: {
      patientId: `DEMO-PT-${Date.now()}`,
      name: isDemoTumor ? 'Demo Patient (Tumor Sample)' : 'Demo Patient (Normal Sample)',
      age: 45,
      gender: 'Female'
    },
    sampleType: 'Brain MRI',
    collectionInfo: {
      collectionDate: new Date()
    },
    status: 'Completed',
    priority: 'High',
    images: [{
      filename: copiedFilename,
      originalName: isDemoTumor ? 'Tumor WSI Sample' : 'Normal WSI Sample',
      mimetype: 'image/jpeg',
      size: fs.statSync(demoImagePath).size,
      uploadedAt: new Date(),
      path: copiedPath,
      url: imageUrl,
      mlAnalysis: {
        prediction: mapPrediction(mlResult.predicted_class),
        confidence: mlResult.confidence,
        riskAssessment: mlResult.risk_assessment,
        processingTime: 2500,
        modelVersion: 'EfficientNetB3-BrainTumor-v1.0',
        analyzedAt: new Date(),
        metadata: mlResult
      }
    }],
    aiAnalysis: {
      overallPrediction: mapPrediction(mlResult.predicted_class),
      averageConfidence: mlResult.confidence,
      highRiskCount: mlResult.is_tumor ? 1 : 0,
      analyzedAt: new Date(),
      modelVersion: 'EfficientNetB3-BrainTumor-v1.0',
      aiInsights: {
        totalImages: 1,
        highRiskImages: mlResult.is_tumor ? 1 : 0,
        modelConfidence: mlResult.confidence
      }
    }
  });

  await demoSample.save();
  console.log(`✅ Demo sample created: ${sampleId} (REAL ML Result: ${mlResult.predicted_class} @ ${(mlResult.confidence * 100).toFixed(1)}%)`);

  res.status(200).json({
    success: true,
    message: `Demo ${demoType} sample analyzed successfully with REAL ML prediction`,
    data: {
      sample: demoSample,
      mlAnalysis: {
        imagesAnalyzed: 1,
        predictions: [mlResult],
        aiInsights: {
          totalImages: 1,
          highRiskImages: mlResult.is_tumor ? 1 : 0,
          modelConfidence: mlResult.confidence
        }
      }
    }
  });
}));

export default router;
