import mongoose from 'mongoose';

const sampleSchema = new mongoose.Schema({
  sampleId: {
    type: String,
    required: true,
    unique: true,
    trim: true
  },
  patientInfo: {
    patientId: {
      type: String,
      required: [true, 'Patient ID is required'],
      trim: true
    },
    name: {
      type: String,
      required: [true, 'Patient name is required'],
      trim: true
    },
    age: {
      type: Number,
      required: [true, 'Patient age is required'],
      min: [0, 'Age cannot be negative'],
      max: [150, 'Age seems unrealistic']
    },
    gender: {
      type: String,
      enum: ['Male', 'Female', 'Other'],
      required: [true, 'Gender is required']
    },
    dateOfBirth: Date,
    contactNumber: String,
    referringPhysician: String,
    medicalHistory: [String],
    currentMedications: [String],
    allergies: [String]
  },
  sampleType: {
    type: String,
    enum: {
      values: ['Brain MRI', 'Chest X-ray', 'Other'],
      message: 'Sample type must be one of: Brain MRI, Chest X-ray, Other'
    },
    required: [true, 'Sample type is required']
  },
  specimenDetails: {
    organ: String,
    site: String,
    size: String,
    color: String,
    consistency: String,
    fixative: {
      type: String,
      default: 'Formalin'
    },
    processingMethod: String,
    staining: {
      type: [String],
      default: ['H&E']
    }
  },
  clinicalInfo: {
    clinicalDiagnosis: String,
    symptoms: [String],
    duration: String,
    medicalHistory: String,
    urgency: {
      type: String,
      enum: ['Routine', 'Urgent', 'STAT'],
      default: 'Routine'
    },
    requestingPhysician: {
      name: String,
      department: String,
      contactInfo: String
    }
  },
  collectionInfo: {
    collectionDate: {
      type: Date,
      default: Date.now
    },
    collectionTime: String,
    collectedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    collectionMethod: String,
    transportConditions: String
  },
  images: [{
    filename: String,
    originalName: String,
    mimetype: String,
    size: Number,
    path: String,
    url: String,
    isDicom: Boolean,
    originalDicomPath: String,
    uploadedAt: { type: Date, default: Date.now },
    magnification: String,
    staining: String,
    fieldOfView: String,
    uploadedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    // ── Grad-CAM heatmap ─────────────────────────────────────────────────
    heatmap: {
      base64:       mongoose.Schema.Types.Mixed, // data:image/png;base64,... string
      type:         String,   // 'gradcam'
      colormap:     String,   // 'jet'
      affectedAreaPct: Number,
      severity:     String
    },
    // ── ML Analysis ───────────────────────────────────────────────────────
    mlAnalysis: {
      prediction: {
        type: String,
        enum: [
          'benign', 'malignant', 'indeterminate',
          'Pneumonia', 'Normal',
          'Glioma', 'Meningioma', 'Pituitary', 'Tumor', 'No Tumor',
          // lowercase versions returned by brain tumor API
          'glioma', 'meningioma', 'pituitary', 'notumor'
        ]
      },
      confidence: Number,
      riskAssessment: {
        type: String,
        enum: ['low', 'medium', 'high']
      },
      processingTime: Number,
      imageId: String,
      modelVersion: String,
      analyzedAt: { type: Date, default: Date.now },
      metadata: mongoose.Schema.Types.Mixed,
      // Pneumonia-specific
      isPneumonia: Boolean,
      severity: String,
      affectedAreaPct: Number,
      confidenceTier: mongoose.Schema.Types.Mixed,
      classProbabilities: mongoose.Schema.Types.Mixed,
      // Brain tumor-specific
      tumorType: String,
      tumorTypeInfo: mongoose.Schema.Types.Mixed,
      // Blood analysis
      bloodAnalysis: {
        malaria: {
          status: String,
          confidence: Number,
          isParasitized: Boolean,
          probabilities: mongoose.Schema.Types.Mixed
        },
        cellCount: {
          platelets: Number,
          rbc: Number,
          wbc: Number,
          totalCells: Number,
          status: String
        }
      }
    }
  }],

  aiAnalysis: {
    overallPrediction: {
      type: String,
      enum: ['benign', 'malignant', 'indeterminate', 'Pneumonia', 'Normal', 'Glioma', 'Meningioma', 'Pituitary', 'Tumor', 'No Tumor']
    },
    averageConfidence: Number,
    highRiskImages: Number,
    totalImagesAnalyzed: Number,
    recommendations: [String],
    flaggedFindings: [String],
    batchAnalyzedAt: Date,
    modelInfo: {
      name: String,
      version: String,
      accuracy: Number
    }
  },
  status: {
    type: String,
    enum: {
      values: ['Received', 'Processing', 'Ready for Analysis', 'Under Review', 'Completed', 'Rejected', 'Archived'],
      message: 'Status must be one of the predefined values'
    },
    default: 'Received'
  },
  priority: {
    type: String,
    enum: ['Low', 'Normal', 'High', 'Critical'],
    default: 'Normal'
  },
  workflow: {
    receivedAt: { type: Date, default: Date.now },
    processedAt: Date,
    analyzedAt: Date,
    reviewedAt: Date,
    completedAt: Date,
    estimatedCompletionTime: Date,
    actualProcessingTime: Number // in minutes
  },
  assignedTo: {
    processor: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    analyst: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    reviewer: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }
  },
  qualityControl: {
    adequacy: {
      type: String,
      enum: ['Adequate', 'Inadequate', 'Limited'],
      default: 'Adequate'
    },
    technicalQuality: {
      type: String,
      enum: ['Excellent', 'Good', 'Fair', 'Poor']
    },
    artifacts: [String],
    notes: String
  },
  tags: [String],
  notes: [String],
  isDeleted: {
    type: Boolean,
    default: false
  },
  deletedAt: Date,
  deletedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for processing time
sampleSchema.virtual('processingDuration').get(function() {
  if (this.workflow.completedAt && this.workflow.receivedAt) {
    return Math.floor((this.workflow.completedAt - this.workflow.receivedAt) / (1000 * 60)); // in minutes
  }
  return null;
});

// Virtual for turnaround time
sampleSchema.virtual('turnaroundTime').get(function() {
  if (this.workflow.completedAt && this.collectionInfo.collectionDate) {
    return Math.floor((this.workflow.completedAt - this.collectionInfo.collectionDate) / (1000 * 60 * 60)); // in hours
  }
  return null;
});

// Indexes for performance
sampleSchema.index({ sampleId: 1 });
sampleSchema.index({ 'patientInfo.patientId': 1 });
sampleSchema.index({ status: 1 });
sampleSchema.index({ sampleType: 1 });
sampleSchema.index({ priority: 1 });
sampleSchema.index({ createdAt: -1 });
sampleSchema.index({ 'collectionInfo.collectionDate': -1 });
sampleSchema.index({ isDeleted: 1 });

// Pre-save middleware to generate sample ID
sampleSchema.pre('save', async function(next) {
  if (!this.sampleId) {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    // Find the count of samples created today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    const count = await this.constructor.countDocuments({
      createdAt: { $gte: today, $lt: tomorrow }
    });
    
    const sequence = String(count + 1).padStart(4, '0');
    this.sampleId = `RDX-${year}${month}${day}-${sequence}`;
  }
  next();
});

// Static method to get sample statistics
sampleSchema.statics.getStats = async function(startDate, endDate) {
  const matchStage = {};
  if (startDate || endDate) {
    matchStage.createdAt = {};
    if (startDate) matchStage.createdAt.$gte = new Date(startDate);
    if (endDate) matchStage.createdAt.$lte = new Date(endDate);
  }

  return await this.aggregate([
    { $match: matchStage },
    {
      $group: {
        _id: '$status',
        count: { $sum: 1 },
        avgProcessingTime: { $avg: '$workflow.actualProcessingTime' }
      }
    },
    {
      $group: {
        _id: null,
        totalSamples: { $sum: '$count' },
        statusBreakdown: {
          $push: {
            status: '$_id',
            count: '$count',
            avgProcessingTime: '$avgProcessingTime'
          }
        }
      }
    }
  ]);
};

// Static method to find samples by patient
sampleSchema.statics.findByPatient = function(patientId) {
  return this.find({ 
    'patientInfo.patientId': patientId,
    isDeleted: false 
  }).sort({ createdAt: -1 });
};

// Static method to find pending samples
sampleSchema.statics.findPending = function() {
  return this.find({
    status: { $in: ['Received', 'Processing', 'Ready for Analysis', 'Under Review'] },
    isDeleted: false
  }).sort({ priority: -1, createdAt: 1 });
};

// Instance method to update status
sampleSchema.methods.updateStatus = function(newStatus, userId) {
  this.status = newStatus;
  
  const now = new Date();
  switch (newStatus) {
    case 'Processing':
      this.workflow.processedAt = now;
      break;
    case 'Ready for Analysis':
      this.workflow.analyzedAt = now;
      break;
    case 'Under Review':
      this.workflow.reviewedAt = now;
      break;
    case 'Completed':
      this.workflow.completedAt = now;
      if (this.workflow.receivedAt) {
        this.workflow.actualProcessingTime = Math.floor((now - this.workflow.receivedAt) / (1000 * 60));
      }
      break;
  }
  
  return this.save();
};

const Sample = mongoose.model('Sample', sampleSchema);

export default Sample;