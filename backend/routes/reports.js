import express from 'express';
import Report from '../models/Report.js';
import Sample from '../models/Sample.js';
import User from '../models/User.js';
import { catchAsync } from '../middleware/errorHandler.js';
import { verifyToken as auth } from '../middleware/auth.js';
import { reportValidations } from '../middleware/validation.js';
import { validationResult } from 'express-validator';

// AI Service for clinical summary generation (Gemini or OpenAI)
let geminiService = null;
try {
  geminiService = await import('../services/geminiService.js');
} catch (error) {
  console.log('ℹ️ AI summary service not available - using fallback summaries');
}

const router = express.Router();

// Generate comprehensive report for a sample
router.post('/generate/:sampleId',
  // auth, // Temporarily disabled for testing
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;
    const { reportType = 'Preliminary', includeImages = true } = req.body;

    console.log('🔄 Generating report for sample:', sampleId);

    // Find the sample with ML analysis
    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Generate report ID
    const reportCount = await Report.countDocuments();
    const reportId = `RPT-${new Date().getFullYear()}-${String(reportCount + 1).padStart(4, '0')}`;

    // Aggregate ML analysis from all images
    const mlAnalysisSummary = sample.images.reduce((summary, image) => {
      if (image.mlAnalysis) {
        summary.totalImages++;
        if (image.mlAnalysis.prediction === 'malignant') {
          summary.malignantDetections++;
        }
        summary.averageConfidence += (image.mlAnalysis.confidence || 0);

        // Map risk assessment to numeric score
        const riskScore = image.mlAnalysis.riskAssessment === 'high' ? 0.8 :
          image.mlAnalysis.riskAssessment === 'medium' ? 0.5 : 0.2;
        summary.maxRiskScore = Math.max(summary.maxRiskScore, riskScore);

        // Collect all detected features
        if (image.mlAnalysis.metadata?.detected_features) {
          summary.allFeatures.push(...image.mlAnalysis.metadata.detected_features);
        }
      }
      return summary;
    }, {
      totalImages: 0,
      malignantDetections: 0,
      averageConfidence: 0,
      maxRiskScore: 0,
      allFeatures: []
    });

    if (mlAnalysisSummary.totalImages > 0) {
      mlAnalysisSummary.averageConfidence /= mlAnalysisSummary.totalImages;
      mlAnalysisSummary.uniqueFeatures = [...new Set(mlAnalysisSummary.allFeatures)];
    }

    // Generate AI interpretation
    const aiInterpretation = generateAIInterpretation(sample, mlAnalysisSummary);

    // Create the report
    const reportData = {
      reportId,
      sampleId: sample._id,
      patientInfo: {
        patientId: sample.patientInfo?.patientId || 'Unknown',
        name: sample.patientInfo?.name || 'Unknown',
        age: sample.patientInfo?.age || 0,
        gender: sample.patientInfo?.gender || 'Unknown',
        contactNumber: sample.patientInfo?.contactNumber || '',
        email: sample.patientInfo?.email || ''
      },
      clinicalInfo: {
        orderingPhysician: sample.clinicalInfo?.orderingPhysician || '',
        clinicalHistory: sample.clinicalInfo?.clinicalHistory || '',
        provisionalDiagnosis: sample.clinicalInfo?.provisionalDiagnosis || sample.clinicalInfo?.clinicalDiagnosis || '',
        sampleSite: sample.specimenDetails?.site || '',
        sampleDate: sample.collectionInfo?.collectionDate || sample.createdAt
      },
      aiAnalysis: {
        totalImages: mlAnalysisSummary.totalImages,
        processedImages: mlAnalysisSummary.totalImages,
        malignantDetections: mlAnalysisSummary.malignantDetections,
        averageConfidence: mlAnalysisSummary.averageConfidence,
        maxRiskScore: mlAnalysisSummary.maxRiskScore,
        detectedFeatures: mlAnalysisSummary.uniqueFeatures,
        overallRisk: (() => {
          // Calculate based on tumor probability (confidence) like other endpoints
          const firstImg = sample.images?.[0];
          const conf = firstImg?.mlAnalysis?.confidence || 0;
          const tumorProb = conf <= 1 ? conf * 100 : conf;
          return tumorProb >= 70 ? 'High' : tumorProb >= 50 ? 'Medium' : 'Low';
        })(),
        aiInterpretation
      },
      imageAnalysis: includeImages ? sample.images.map(img => ({
        filename: img.filename,
        originalName: img.originalName,
        mlAnalysis: img.mlAnalysis,
        staining: img.staining,
        magnification: img.magnification
      })) : [],
      reportType,
      analysisId: sample._id, // Use sample ID as analysis reference
      status: 'draft',
      generatedBy: req.user?.id || 'system', // Handle case when no user auth
      generatedAt: new Date(),
      version: '1.0'
    };

    const report = new Report(reportData);
    await report.save();

    // Update sample status to 'Under Review'
    sample.status = 'Under Review';
    sample.lastUpdated = new Date();

    // Initialize history array if it doesn't exist
    if (!sample.history) {
      sample.history = [];
    }

    sample.history.push({
      status: 'Under Review',
      timestamp: new Date(),
      updatedBy: req.user?.id || 'system',
      notes: 'Report generated automatically based on AI analysis'
    });
    await sample.save();

    res.json({
      success: true,
      message: 'Report generated successfully',
      data: {
        reportId: report.reportId,
        _id: report._id,
        status: report.status,
        aiAnalysis: report.aiAnalysis,
        generatedAt: report.generatedAt
      }
    });
  })
);

// Get report by ID
router.get('/:reportId',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;

    const report = await Report.findOne({
      $or: [
        { reportId },
        { _id: reportId }
      ]
    }).populate('generatedBy', 'name email role');

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    res.json({
      success: true,
      data: report
    });
  })
);

// Get all reports with pagination and filtering
router.get('/',
  auth,
  catchAsync(async (req, res) => {
    const {
      page = 1,
      limit = 10,
      status,
      reportType,
      dateFrom,
      dateTo,
      search
    } = req.query;

    const filter = {};

    if (status) filter.status = status;
    if (reportType) filter.reportType = reportType;

    if (dateFrom || dateTo) {
      filter.generatedAt = {};
      if (dateFrom) filter.generatedAt.$gte = new Date(dateFrom);
      if (dateTo) filter.generatedAt.$lte = new Date(dateTo);
    }

    if (search) {
      filter.$or = [
        { reportId: { $regex: search, $options: 'i' } },
        { 'patientInfo.name': { $regex: search, $options: 'i' } },
        { 'patientInfo.patientId': { $regex: search, $options: 'i' } }
      ];
    }

    const options = {
      page: parseInt(page),
      limit: parseInt(limit),
      sort: { generatedAt: -1 },
      populate: {
        path: 'generatedBy',
        select: 'name email role'
      }
    };

    const reports = await Report.paginate(filter, options);

    res.json({
      success: true,
      data: reports
    });
  })
);

// Update report status (draft -> review -> approved -> finalized)
router.patch('/:reportId/status',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;
    const { status, reviewNotes } = req.body;

    const validStatuses = ['draft', 'review', 'approved', 'finalized', 'rejected'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid status. Must be one of: ' + validStatuses.join(', ')
      });
    }

    const report = await Report.findOne({
      $or: [{ reportId }, { _id: reportId }]
    });

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    // Add to workflow history
    report.workflow.push({
      status,
      timestamp: new Date(),
      updatedBy: req.user.id,
      notes: reviewNotes
    });

    report.status = status;
    report.lastUpdated = new Date();

    if (status === 'finalized') {
      report.finalizedAt = new Date();
      report.finalizedBy = req.user.id;
    }

    await report.save();

    res.json({
      success: true,
      message: `Report ${status} successfully`,
      data: {
        reportId: report.reportId,
        status: report.status,
        lastUpdated: report.lastUpdated
      }
    });
  })
);

// Download report as PDF (placeholder for now)
router.get('/:reportId/download',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;
    const { format = 'pdf' } = req.query;

    const report = await Report.findOne({
      $or: [{ reportId }, { _id: reportId }]
    });

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    // For now, return report data (in production, generate actual PDF)
    res.json({
      success: true,
      message: 'Report download ready',
      data: {
        reportId: report.reportId,
        format,
        downloadUrl: `/api/reports/${reportId}/pdf`, // Future implementation
        report
      }
    });
  })
);

// Generate AI clinical summary using Gemini
router.post('/generate-summary/:sampleId',
  // auth, // Temporarily disabled for testing
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('🤖 Generating AI clinical summary for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Aggregate ML results
    const mlResults = {
      totalImages: sample.images?.length || 0,
      malignantDetections: 0,
      averageConfidence: 0,
      overallRisk: 'Unknown',
      tumorProbability: null,
      sampleType: sample.sampleType || sample.imageType || 'Brain MRI'
    };

    if (sample.images?.length > 0) {
      let totalConfidence = 0;
      let maxRiskScore = 0;

      sample.images.forEach(img => {
        if (img.mlAnalysis) {
          if (img.mlAnalysis.isPneumonia || img.mlAnalysis.is_tumor ||
              img.mlAnalysis.metadata?.is_tumor ||
              img.mlAnalysis.prediction === 'malignant' ||
              img.mlAnalysis.prediction === 'Pneumonia') {
            mlResults.malignantDetections++;
          }
          totalConfidence += img.mlAnalysis.confidence || 0;

          const riskScore = img.mlAnalysis.riskAssessment === 'high' ? 0.8 :
            img.mlAnalysis.riskAssessment === 'medium' ? 0.5 : 0.2;
          maxRiskScore = Math.max(maxRiskScore, riskScore);

          // Extract tumor probability from multiple possible locations
          let rawTumorProb =
            img.mlAnalysis.metadata?.probabilities?.tumor ||
            img.mlAnalysis.metadata?.probabilities?.Tumor ||
            img.mlAnalysis.probabilities?.tumor ||
            (img.mlAnalysis.metadata?.is_tumor ? img.mlAnalysis.confidence : null) ||
            (img.mlAnalysis.prediction === 'malignant' ? img.mlAnalysis.confidence : null) ||
            0;

          if (rawTumorProb > 0) {
            mlResults.tumorProbability = rawTumorProb;
          }
        }
      });

      mlResults.averageConfidence = totalConfidence / sample.images.length;
      mlResults.overallRisk = maxRiskScore > 0.7 ? 'High' : maxRiskScore > 0.4 ? 'Moderate' : 'Low';
    }

    // Generate summary using Gemini or fallback
    let summary;
    if (geminiService?.generateClinicalSummary) {
      summary = await geminiService.generateClinicalSummary(mlResults, sample.patientInfo || {});
    } else {
      // Fallback summary
      const hasMalignant = mlResults.malignantDetections > 0;
      const confidence = (mlResults.averageConfidence * 100).toFixed(1);

      summary = {
        success: true,
        summary: hasMalignant
          ? `AI analysis detected ${mlResults.malignantDetections} malignant region(s). Risk: ${mlResults.overallRisk}. Confidence: ${confidence}%. Further evaluation recommended.`
          : `AI analysis shows no significant abnormalities. Risk: ${mlResults.overallRisk}. Confidence: ${confidence}%. Normal findings.`,
        generatedBy: 'Fallback',
        timestamp: new Date().toISOString()
      };
    }

    res.json({
      success: true,
      data: {
        sampleId,
        mlResults,
        clinicalSummary: summary
      }
    });
  })
);

// Generate FULL AI Report using Gemini
router.post('/generate-full/:sampleId',
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('🤖 Generating FULL AI report for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Build ML results object
    const mlResults = {
      totalImages: sample.images?.length || 0,
      isPositive: false,
      tumorProbability: 0,
      confidence: 0,
      riskLevel: 'Low',
      sampleType: sample.sampleType || sample.imageType || 'Brain MRI'
    };

    if (sample.images?.length > 0) {
      const firstImage = sample.images[0];
      if (firstImage?.mlAnalysis) {
        const analysis = firstImage.mlAnalysis;

        // Model's confidence IS the tumor probability
        const rawConfidence = analysis.confidence || 0;
        const confidencePercent = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;

        // If explicit probabilities exist, prefer them
        let tumorProbPercent = confidencePercent;
        if (analysis.metadata?.probabilities?.tumor !== undefined) {
          const prob = analysis.metadata.probabilities.tumor;
          tumorProbPercent = prob <= 1 ? prob * 100 : prob;
        }

        mlResults.tumorProbability = tumorProbPercent;
        mlResults.confidence = confidencePercent;
        mlResults.isPositive = tumorProbPercent >= 54;
        mlResults.riskLevel = tumorProbPercent >= 70 ? 'High' : tumorProbPercent >= 50 ? 'Medium' : 'Low';
      }
    }

    // Generate full report using AI - SINGLE REQUEST to avoid rate limits
    const isPneumoniaReport = mlResults.sampleType === 'Chest X-ray' || sample.imageType === 'pneumonia';
    let reportContent;
    if (geminiService?.generateFullReport) {
      console.log('🚀 Making SINGLE AI API request...');
      reportContent = await geminiService.generateFullReport(sample, mlResults);
    } else {
      // Fallback content — type-aware
      reportContent = {
        success: true,
        generatedBy: 'Fallback',
        content: {
          clinicalSummary: isPneumoniaReport
            ? (mlResults.isPositive
              ? `AI analysis detected pneumonia indicators with ${mlResults.confidence.toFixed(1)}% confidence. DenseNet121 + EfficientNet-B0 ensemble model flagged abnormal opacities.`
              : `AI analysis confirms normal chest X-ray with ${mlResults.confidence.toFixed(1)}% confidence. No pneumonia indicators detected.`)
            : (mlResults.isPositive
              ? `AI analysis detected potential brain tumor with ${mlResults.confidence.toFixed(1)}% confidence. EfficientNetB3 model flagged abnormal features in MRI scan.`
              : `AI analysis confirms normal brain MRI with ${mlResults.confidence.toFixed(1)}% confidence. No tumor detected.`),
          interpretation: isPneumoniaReport
            ? 'AI-powered chest X-ray analysis has been completed using DenseNet121 + EfficientNet-B0 ensemble. Results should be correlated with clinical symptoms and patient history.'
            : 'AI-powered brain MRI analysis has been completed using EfficientNetB3 4-class classifier. Results should be correlated with clinical symptoms and neurological examination.',
          recommendations: isPneumoniaReport
            ? (mlResults.isPositive
              ? ['Chest CT scan recommended for further evaluation', 'Sputum culture and sensitivity testing advised', 'Consider blood work (CBC, CRP)', 'Follow-up chest X-ray in 4-6 weeks']
              : ['No pneumonia detected — routine monitoring', 'Follow up if symptoms persist', 'Clinical correlation advised'])
            : (mlResults.isPositive
              ? ['Referral to neurosurgery/neuro-oncology recommended', 'Contrast-enhanced MRI for characterization', 'Consider biopsy for confirmation', 'Correlate with neurological exam']
              : ['No tumor detected — routine follow-up', 'Continue periodic screening if risk factors present', 'Clinical correlation advised']),
          morphologicalFindings: isPneumoniaReport
            ? (mlResults.isPositive
              ? 'AI detected opacities suggestive of pulmonary consolidation consistent with pneumonia.'
              : 'Clear lung fields with no consolidation or infiltrates observed.')
            : (mlResults.isPositive
              ? 'AI detected features suggestive of intracranial mass lesion.'
              : 'Normal brain parenchyma with no mass lesion detected.'),
          conclusion: isPneumoniaReport
            ? (mlResults.isPositive
              ? 'Findings suggest pneumonia. Radiologist review and treatment initiation recommended.'
              : 'No pneumonia detected. Routine follow-up recommended if clinically indicated.')
            : (mlResults.isPositive
              ? 'Findings suggest intracranial lesion. Neuroradiologist review and further workup recommended.'
              : 'No brain tumor detected. Routine follow-up recommended.')
        }
      };
    }

    // Use the report content for summary and recommendations (no extra API calls)
    const summary = reportContent?.content?.clinicalSummary ? {
      success: true,
      summary: reportContent.content.clinicalSummary,
      generatedBy: reportContent.generatedBy || 'Gemini AI'
    } : null;

    const recommendations = reportContent?.content?.recommendations ? {
      success: true,
      recommendations: reportContent.content.recommendations,
      generatedBy: reportContent.generatedBy || 'Gemini AI'
    } : null;

    res.json({
      success: true,
      data: {
        sampleId,
        mlResults,
        report: reportContent,
        clinicalSummary: summary,
        recommendations: recommendations
      }
    });
  })
);

// Generate AI interpretation based on ML analysis
function generateAIInterpretation(sample, mlSummary) {
  const interpretations = [];

  if (mlSummary.totalImages === 0) {
    return "No images were available for AI analysis.";
  }

  // Overall assessment
  if (mlSummary.malignantDetections > 0) {
    interpretations.push(`AI analysis identified potential malignant features in ${mlSummary.malignantDetections} out of ${mlSummary.totalImages} images.`);
  } else {
    interpretations.push(`AI analysis did not identify malignant features in the examined ${mlSummary.totalImages} images.`);
  }

  // Confidence assessment
  const avgConfidence = (mlSummary.averageConfidence * 100).toFixed(1);
  interpretations.push(`Average prediction confidence: ${avgConfidence}%.`);

  // Risk assessment
  const riskScore = (mlSummary.maxRiskScore * 100).toFixed(0);
  if (mlSummary.maxRiskScore > 0.7) {
    interpretations.push(`High risk assessment (${riskScore}%) - recommend immediate pathologist review.`);
  } else if (mlSummary.maxRiskScore > 0.4) {
    interpretations.push(`Moderate risk assessment (${riskScore}%) - standard pathologist review recommended.`);
  } else {
    interpretations.push(`Low risk assessment (${riskScore}%) - routine pathologist review.`);
  }

  // Feature analysis
  if (mlSummary.uniqueFeatures.length > 0) {
    interpretations.push(`Detected features include: ${mlSummary.uniqueFeatures.join(', ')}.`);
  }

  interpretations.push("Note: AI analysis is a screening tool and should not replace expert pathologist interpretation.");

  return interpretations.join(' ');
}

// PDF Download endpoint
router.get('/download-pdf/:sampleId',
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('📄 Generating PDF for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      console.log('❌ Sample not found:', sampleId);
      return res.status(404).json({ success: false, message: 'Sample not found' });
    }

    console.log('✅ Sample found:', sample.sampleId || sampleId);

    // Import Puppeteer dynamically
    let puppeteer;
    try {
      puppeteer = await import('puppeteer');
      console.log('✅ Puppeteer imported');
    } catch (error) {
      console.error('❌ Failed to import Puppeteer:', error.message);
      return res.status(500).json({
        success: false,
        message: 'PDF generation unavailable - Puppeteer not installed',
        error: error.message
      });
    }

    try {
      // Build report data
      const patientInfo = sample.patientInfo || {};
      const firstImage = sample.images?.[0];
      const analysis = firstImage?.mlAnalysis || {};

      // Determine if this is a pneumonia or brain tumor report
      const isPneumonia = sample.sampleType === 'Chest X-ray' || sample.imageType === 'pneumonia';
      const reportTitle = isPneumonia ? 'CHEST X-RAY REPORT' : 'NEURORADIOLOGY REPORT';
      const probLabel = isPneumonia ? 'Pneumonia Probability' : 'Tumor Probability';

      // Extract confidence and probability
      const rawConfidence = analysis.confidence || 0;
      const confidencePercent = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;

      let detectionProb = confidencePercent;

      if (!isPneumonia && analysis.metadata?.probabilities?.tumor !== undefined) {
        detectionProb = analysis.metadata.probabilities.tumor <= 1
          ? analysis.metadata.probabilities.tumor * 100
          : analysis.metadata.probabilities.tumor;
      }

      const probValue = detectionProb.toFixed(1);
      const confidence = confidencePercent.toFixed(1);

      // Determine if positive
      const isPositive = isPneumonia
        ? (analysis.isPneumonia || analysis.prediction === 'malignant')
        : parseFloat(probValue) >= 54;

      // Risk level
      const riskLevel = isPneumonia
        ? (isPositive ? 'High' : 'Low')
        : (parseFloat(probValue) >= 70 ? 'High' : parseFloat(probValue) >= 50 ? 'Medium' : 'Low');

      // Model info
      const modelName = isPneumonia ? 'DenseNet121 + EfficientNet-B0 Ensemble' : 'EfficientNetB3';

      console.log('📊 PDF Data:', { isPneumonia, probValue, confidence, isPositive, riskLevel });

      // Generate report HTML
      const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Arial', sans-serif; 
      padding: 40px;
      color: #1a1a1a;
      line-height: 1.6;
    }
    .header {
      border-bottom: 4px solid #1e40af;
      padding-bottom: 20px;
      margin-bottom: 30px;
    }
    .header h1 {
      color: #1e40af;
      font-size: 28px;
      margin-bottom: 10px;
    }
    .header .info {
      display: flex;
      justify-content: space-between;
      margin-top: 15px;
      font-size: 12px;
      color: #666;
    }
    .section {
      margin-bottom: 25px;
      page-break-inside: avoid;
    }
    .section-title {
      background: #f3f4f6;
      padding: 10px 15px;
      font-size: 16px;
      font-weight: bold;
      color: #1e40af;
      margin-bottom: 15px;
      border-left: 4px solid #1e40af;
    }
    .field-group {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin-bottom: 15px;
    }
    .field {
      padding: 10px;
      background: #f9fafb;
      border-radius: 4px;
    }
    .field-label {
      font-size: 11px;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 5px;
    }
    .field-value {
      font-size: 14px;
      color: #1f2937;
      font-weight: 500;
    }
    .alert {
      padding: 15px;
      border-radius: 6px;
      margin: 15px 0;
    }
    .alert-positive {
      background: #fef2f2;
      border-left: 4px solid #dc2626;
      color: #991b1b;
    }
    .alert-negative {
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
      color: #166534;
    }
    .alert-title {
      font-weight: bold;
      font-size: 16px;
      margin-bottom: 8px;
    }
    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 2px solid #e5e7eb;
      font-size: 11px;
      color: #6b7280;
      text-align: center;
    }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 8px;
    }
    .badge-high { background: #fee2e2; color: #991b1b; }
    .badge-low { background: #dcfce7; color: #166534; }
  </style>
</head>
<body>
  <div class="header">
    <h1>DIAGNOSTIC REPORT</h1>
    <div class="info">
      <div>
        <strong>RecursiaDx Digital Pathology Lab</strong><br>
        Email: lab@recursiadx.com | Phone: +91 99999 88888
      </div>
      <div style="text-align: right;">
        <strong>Report Type:</strong> ${reportTitle}<br>
        <strong>Report ID:</strong> ${sample.sampleId || sample._id}<br>
        <strong>Date:</strong> ${new Date().toLocaleDateString()}
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Patient Information</div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Patient Name</div>
        <div class="field-value">${patientInfo.name || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Patient ID</div>
        <div class="field-value">${patientInfo.id || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Age</div>
        <div class="field-value">${patientInfo.age || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Gender</div>
        <div class="field-value">${patientInfo.gender || 'N/A'}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Sample Information</div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Sample Type</div>
        <div class="field-value">${sample.sampleType || 'Brain MRI'}</div>
      </div>
      <div class="field">
        <div class="field-label">Collection Date</div>
        <div class="field-value">${sample.sampleInfo?.collectionDate || new Date().toLocaleDateString()}</div>
      </div>
      <div class="field">
        <div class="field-label">Images Analyzed</div>
        <div class="field-value">${sample.images?.length || 0}</div>
      </div>
      <div class="field">
        <div class="field-label">Analysis Status</div>
        <div class="field-value">${sample.status || 'Completed'}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">AI Analysis Results</div>
    <div class="alert ${isPositive ? 'alert-positive' : 'alert-negative'}">
      <div class="alert-title">${isPositive ? '⚠️ Abnormal Findings' : '✓ Normal Findings'}</div>
      <p>
        ${isPneumonia
          ? (isPositive
            ? `AI analysis detected pneumonia indicators with ${confidence}% confidence. ${probLabel}: ${probValue}%. Risk level: ${riskLevel}. Model: ${modelName}.`
            : `AI analysis confirms normal chest X-ray with ${confidence}% confidence. No pneumonia detected. Model: ${modelName}.`)
          : (isPositive
            ? `AI analysis detected potential brain tumor with ${confidence}% confidence. ${probLabel}: ${probValue}%. Risk level: ${riskLevel}. Model: ${modelName}.`
            : `AI analysis confirms normal brain MRI with ${confidence}% confidence. No tumor detected. Model: ${modelName}.`)
        }
      </p>
    </div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Detection Result</div>
        <div class="field-value">${isPositive ? 'POSITIVE (Abnormal)' : 'NEGATIVE (Normal)'}</div>
      </div>
      <div class="field">
        <div class="field-label">AI Confidence</div>
        <div class="field-value">${confidence}%</div>
      </div>
      <div class="field">
        <div class="field-label">${probLabel}</div>
        <div class="field-value">${probValue}%</div>
      </div>
      <div class="field">
        <div class="field-label">Risk Assessment</div>
        <div class="field-value">
          <span class="badge ${riskLevel?.includes('High') ? 'badge-high' : 'badge-low'}">
            ${riskLevel}
          </span>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Clinical Recommendations</div>
    <ul style="padding-left: 20px; margin-top: 10px;">
      ${isPneumonia
        ? (isPositive ? `
          <li style="margin-bottom: 8px;">Chest CT scan recommended for further evaluation</li>
          <li style="margin-bottom: 8px;">Sputum culture and sensitivity testing advised</li>
          <li style="margin-bottom: 8px;">Consider clinical correlation with blood work (CBC, CRP)</li>
          <li style="margin-bottom: 8px;">Follow-up chest X-ray in 4-6 weeks to confirm resolution</li>
        ` : `
          <li style="margin-bottom: 8px;">No pneumonia detected — no immediate follow-up required</li>
          <li style="margin-bottom: 8px;">Continue routine monitoring if symptoms persist</li>
          <li style="margin-bottom: 8px;">Consult with ordering physician for clinical correlation</li>
        `)
        : (isPositive ? `
          <li style="margin-bottom: 8px;">Referral to neurosurgery/neuro-oncology recommended</li>
          <li style="margin-bottom: 8px;">Contrast-enhanced MRI for detailed characterization</li>
          <li style="margin-bottom: 8px;">Consider biopsy for histopathological confirmation</li>
          <li style="margin-bottom: 8px;">Correlate with clinical symptoms and neurological examination</li>
        ` : `
          <li style="margin-bottom: 8px;">No tumor detected — routine follow-up as clinically indicated</li>
          <li style="margin-bottom: 8px;">Continue periodic screening if risk factors are present</li>
          <li style="margin-bottom: 8px;">Consult with ordering physician for clinical correlation</li>
        `)
      }
    </ul>
  </div>

  <div class="footer">
    <p><strong>RecursiaDx AI-Powered Digital Pathology Platform</strong></p>
    <p>This report was generated using AI-assisted analysis (${modelName}). All findings should be confirmed by a qualified ${isPneumonia ? 'radiologist' : 'neuroradiologist'}.</p>
    <p>HIPAA Compliant | Real-time AI Processing | For professional use only</p>
  </div>
</body>
</html>
      `;

      // Launch Puppeteer
      const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
      });

      const page = await browser.newPage();
      await page.setContent(html, { waitUntil: 'networkidle0' });

      // Generate PDF
      const pdf = await page.pdf({
        format: 'A4',
        printBackground: true,
        margin: {
          top: '20px',
          right: '20px',
          bottom: '20px',
          left: '20px'
        }
      });

      await browser.close();

      // Send PDF
      res.contentType('application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="RecursiaDx_Report_${sample.sampleId || sampleId}.pdf"`);
      res.end(pdf, 'binary');

      console.log('✅ PDF generated successfully');

    } catch (error) {
      console.error('❌ PDF generation error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to generate PDF',
        error: error.message
      });
    }
  })
);

export default router;