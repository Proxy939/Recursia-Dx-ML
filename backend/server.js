// Load environment variables FIRST (before any other imports)
import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import fs from 'fs';
import { execSync } from 'child_process';

// Import routes AFTER dotenv.config() so API keys are available
import authRoutes from './routes/auth.js';
import sampleRoutes from './routes/samples.js';
import reportRoutes from './routes/reports.js';

// Import middleware
import {
  globalErrorHandler,
  notFound,
  requestLogger,
  securityHeaders,
  corsErrorHandler,
  timeoutHandler
} from './middleware/errorHandler.js';

// ES6 module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Auto-heatmap generation function (copied from samples.js)
async function generateAutoHeatmap(imagePath, imageId) {
  return new Promise((resolve) => {
    try {
      const heatmapExamplesDir = path.join(process.cwd(), '..', 'ml', 'heatmap_examples');
      const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

      // Ensure output directory exists
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Available heatmap examples with their analytics
      const availableHeatmaps = [
        {
          source: 'heatmap_hot.png',
          type: 'tumor_probability',
          colormap: 'hot',
          analytics: {
            min_value: 0.0,
            max_value: 0.9073608271482184,
            mean_value: 0.28915497976794247,
            std_value: 0.234978657554125,
            shape: [16, 16],
            hotspots: 17,
            total_pixels: 256
          }
        },
        {
          source: 'heatmap_viridis.png',
          type: 'confidence',
          colormap: 'viridis',
          analytics: {
            min_value: 0.3768195980284063,
            max_value: 0.9339579326308897,
            mean_value: 0.6236872365395125,
            std_value: 0.1339540600255726,
            shape: [16, 16],
            hotspots: 85,
            total_pixels: 256
          }
        },
        {
          source: 'heatmap_plasma.png',
          type: 'risk_score',
          colormap: 'plasma',
          analytics: {
            min_value: 0.2685816582514944,
            max_value: 0.8378640996056985,
            mean_value: 0.5513551464485202,
            std_value: 0.1452114761049256,
            shape: [16, 16],
            hotspots: 42,
            total_pixels: 256
          }
        }
      ];

      // Select a random heatmap or rotate based on imageId
      const heatmapIndex = Math.floor(Math.random() * availableHeatmaps.length);
      const selectedHeatmap = availableHeatmaps[heatmapIndex];

      const timestamp = Date.now();
      const outputFilename = `auto_heatmap_${imageId}_${timestamp}.png`;
      const sourcePath = path.join(heatmapExamplesDir, selectedHeatmap.source);
      const outputPath = path.join(outputDir, outputFilename);

      console.log(`🎨 Extracting heatmap from examples: ${selectedHeatmap.source} -> ${outputFilename}`);

      // Copy the example heatmap to the output location
      if (fs.existsSync(sourcePath)) {
        fs.copyFileSync(sourcePath, outputPath);

        // Read the heatmap file and convert to base64
        let heatmapBase64 = null;
        try {
          const imageBuffer = fs.readFileSync(outputPath);
          heatmapBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
        } catch (e) {
          console.error('Failed to convert heatmap to base64:', e.message);
        }

        console.log(`✅ Heatmap extracted from examples: ${outputFilename}`);

        resolve({
          success: true,
          heatmap: {
            filename: outputFilename,
            path: `/api/samples/heatmap/${outputFilename}`,
            base64: heatmapBase64, // Add base64 encoded image
            type: selectedHeatmap.type,
            colormap: selectedHeatmap.colormap,
            analytics: selectedHeatmap.analytics
          }
        });
      } else {
        console.error(`❌ Source heatmap not found: ${sourcePath}`);
        resolve({ success: false, error: `Source heatmap not found: ${selectedHeatmap.source}` });
      }

    } catch (error) {
      console.error(`❌ Auto-heatmap extraction error for image ${imageId}:`, error.message);
      resolve({ success: false, error: error.message });
    }
  });
}

// Debug: Check if OpenAI API key is loaded
console.log('🔑 OpenAI API Key loaded:', process.env.OPENAI_API_KEY ? '✅ YES' : '❌ NO');

const app = express();

// Security middleware
app.use(helmet({
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: (process.env.RATE_LIMIT_WINDOW || 15) * 60 * 1000, // 15 minutes
  max: process.env.RATE_LIMIT_MAX_REQUESTS || 100, // limit each IP to 100 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'http://localhost:3000,http://localhost:5173').split(',');

    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
};

app.use(cors(corsOptions));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
app.use(requestLogger);

// DICOM-to-PNG middleware: intercept .dcm file requests, convert to PNG on-the-fly

app.use('/uploads', (req, res, next) => {
  // Only intercept .dcm file requests
  if (!req.path.toLowerCase().endsWith('.dcm')) {
    return next();
  }

  const dcmPath = path.join(__dirname, 'uploads', req.path);
  const pngPath = dcmPath.replace(/\.dcm$/i, '.png');

  // Check if PNG already exists (cached conversion)
  if (fs.existsSync(pngPath)) {
    return res.sendFile(pngPath);
  }

  // Check if DCM file exists
  if (!fs.existsSync(dcmPath)) {
    return res.status(404).json({ error: 'File not found' });
  }

  // Convert DICOM to PNG using standalone Python script
  try {
    const dcm2pngScript = path.join(__dirname, '..', 'ml', 'utils', 'dcm2png.py');
    execSync(`python "${dcm2pngScript}" "${dcmPath}" "${pngPath}"`, {
      timeout: 30000,
      stdio: 'pipe'
    });
    console.log(`📸 On-the-fly DICOM → PNG: ${path.basename(pngPath)}`);
    return res.sendFile(pngPath);
  } catch (err) {
    console.error(`⚠️ DICOM conversion failed: ${err.message}`);
    return res.status(500).json({ error: 'Failed to convert DICOM to PNG' });
  }
});

// Static files for uploads
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    // Generate unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    // Allow DICOM files by extension (browsers send them as application/octet-stream)
    const ext = file.originalname.toLowerCase().split('.').pop();
    if (ext === 'dcm') {
      return cb(null, true);
    }
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPEG, PNG, TIFF, BMP, and DICOM (.dcm) files are allowed.'));
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'RecursiaDx API is running',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV,
    version: '1.0.0'
  });
});

// API routes
app.use('/api/auth', authRoutes);

// REMOVED: Legacy /api/upload-simple route — it used Math.random() fake predictions.
// All uploads should go through /api/samples/upload-with-analysis which uses real ML.

// Test route bypassing samples router
app.post('/api/test-no-auth', (req, res) => {
  console.log('🎯 DIRECT TEST ROUTE REACHED!');
  res.json({ success: true, message: 'Direct test works!' });
});

app.use('/api/samples', sampleRoutes);
app.use('/api/reports', reportRoutes);

// API documentation endpoint
app.get('/api', (req, res) => {
  res.json({
    message: 'RecursiaDx Digital Pathology API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      samples: '/api/samples',
      health: '/health'
    },
    documentation: 'https://docs.recursiadx.com'
  });
});

// 404 handler
app.use(notFound);

// Error handling middleware (must be last)
app.use(globalErrorHandler);

// Database connection
const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/recursiadx', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.error('Database connection error:', error.message);
    process.exit(1);
  }
};

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
  console.log('Unhandled Rejection at:', promise, 'reason:', err);
  if (global.server) {
    global.server.close(() => {
      process.exit(1);
    });
  } else {
    process.exit(1);
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.log('Uncaught Exception thrown:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received. Shutting down gracefully...');
  if (global.server) {
    global.server.close(() => {
      console.log('Process terminated');
    });
  }
});

// Connect to database and start server
const PORT = process.env.PORT || 5001;

connectDB().then(() => {
  const server = app.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                     RecursiaDx Backend API                   ║
║══════════════════════════════════════════════════════════════║
║  🚀 Server running on port ${PORT}                              ║
║  🌍 Environment: ${process.env.NODE_ENV || 'development'}                        ║
║  📊 Database: Connected to MongoDB                          ║
║  🔒 Security: Helmet, CORS, Rate Limiting enabled           ║
║  📝 Logging: Request logging active                         ║
╚══════════════════════════════════════════════════════════════╝
    `);
  });

  // Make server available for graceful shutdown
  global.server = server;
});
