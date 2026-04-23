import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Skeleton } from '@/components/ui/skeleton'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { toast } from 'sonner'
import { 
  Brain, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle2, 
  Clock,
  Target,
  PieChart,
  BarChart3
} from 'lucide-react'

export function AnalysisDashboard({ onNext, sample, analysisType = 'general' }) {
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [currentStage, setCurrentStage] = useState('preprocessing')
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const [realTimeData, setRealTimeData] = useState(null)
  const [pneumoniaAnalysisData, setPneumoniaAnalysisData] = useState(null)
  
  // Determine image type from sample (tissue or pneumonia)
  const imageType = sample?.sampleType === 'Chest X-ray' || sample?.imageType === 'pneumonia' ? 'pneumonia' : 'tissue'
  
  // State for active tab (allows manual switching between tabs)
  const [activeTab, setActiveTab] = useState(imageType === 'pneumonia' ? 'pneumonia-analysis' : 'tissue-analysis')
  
  // Update active tab when imageType changes
  useEffect(() => {
    setActiveTab(imageType === 'pneumonia' ? 'pneumonia-analysis' : 'tissue-analysis')
  }, [imageType])

  // Enhanced debugging for sample data
  console.log('🔍 AnalysisDashboard - Sample data received:', sample)
  console.log('🔍 AnalysisDashboard - Sample type:', typeof sample)
  console.log('🔍 AnalysisDashboard - Image type:', imageType)
  console.log('🔍 AnalysisDashboard - Sample keys:', sample ? Object.keys(sample) : [])
  
  if (sample?.images) {
    console.log('🔍 AnalysisDashboard - Images:', sample.images)
    sample.images.forEach((img, idx) => {
      console.log(`🔍 Image ${idx + 1} analysis:`, {
        filename: img.filename,
        hasHeatmap: !!img.heatmap,
        hasMLAnalysis: !!img.mlAnalysis,
        hasPneumoniaAnalysis: !!img.mlAnalysis?.pneumoniaAnalysis,
        pneumoniaAnalysisData: img.mlAnalysis?.pneumoniaAnalysis,
        heatmapStructure: img.heatmap ? Object.keys(img.heatmap) : null
      })
    })
  }

  // Extract real data from sample if available
  const extractRealData = () => {
    if (!sample || !sample.images || sample.images.length === 0) {
      return null
    }

    const images = sample.images
    const totalImages = images.length
    const mlAnalyses = images.filter(img => img.mlAnalysis).map(img => img.mlAnalysis)
    
    if (mlAnalyses.length === 0) {
      return null
    }

    // Aggregate ML results from all images
    // Check for tumor detection - model returns 'Tumor' or uses is_tumor flag
    const tumorDetections = mlAnalyses.filter(analysis => 
      analysis.prediction === 'Tumor' || 
      analysis.prediction === 'malignant' ||
      analysis.is_tumor === true ||
      (analysis.metadata?.prediction?.is_tumor === true)
    )
    const avgConfidence = mlAnalyses.reduce((sum, analysis) => sum + (analysis.confidence || 0), 0) / mlAnalyses.length
    const riskLevels = mlAnalyses.map(analysis => analysis.riskAssessment || 'medium')

    // Calculate tumor percentage
    const tumorPercentage = mlAnalyses.length > 0 ? (tumorDetections.length / mlAnalyses.length) * 100 : 0
    
    // Calculate average tumor probability from all analyses
    // Model's confidence IS the tumor probability
    const avgTumorProbability = mlAnalyses.reduce((sum, analysis) => {
      // Model's confidence is the tumor probability
      let tumorProb = analysis.metadata?.probabilities?.tumor || analysis.confidence || 0
      
      // Convert to percentage if decimal
      if (tumorProb <= 1) tumorProb = tumorProb * 100
      
      console.log('🔍 Tumor prob extraction:', {
        confidence: analysis.confidence,
        metadata_probs: analysis.metadata?.probabilities,
        extracted: tumorProb
      })
      
      return sum + tumorProb
    }, 0) / mlAnalyses.length

    // Get the highest tumor probability for critical assessment
    const maxTumorProbability = Math.max(...mlAnalyses.map(analysis => {
      let prob = analysis.metadata?.probabilities?.tumor || analysis.confidence || 0
      return prob <= 1 ? prob * 100 : prob
    }))

    return {
      totalImages,
      analyzedImages: mlAnalyses.length,
      tumorDetected: tumorDetections.length > 0,
      tumorCount: tumorDetections.length,
      tumorPercentage: tumorPercentage,
      avgTumorProbability: avgTumorProbability, // Already in percentage
      maxTumorProbability: maxTumorProbability, // Already in percentage
      averageConfidence: avgConfidence,
      confidenceRange: {
        min: Math.min(...mlAnalyses.map(a => a.confidence)),
        max: Math.max(...mlAnalyses.map(a => a.confidence))
      },
      riskDistribution: {
        high: riskLevels.filter(r => r && r === 'high').length,
        moderate: riskLevels.filter(r => r && r === 'medium').length,
        low: riskLevels.filter(r => r && r === 'low').length
      },
      highRiskCount: riskLevels.filter(r => r && r === 'high').length,
      patientInfo: sample.patientInfo || {},
      uploadedAt: sample.uploadedAt,
      analyses: mlAnalyses
    }
  }

  // Extract pneumonia analysis data
  const extractPneumoniaAnalysisData = () => {
    console.log('🫁 Extracting pneumonia analysis data...')
    if (!sample || !sample.images || sample.images.length === 0) {
      console.log('❌ No sample or images found')
      return null
    }

    const images = sample.images
    const mlAnalyses = images.filter(img => img.mlAnalysis).map(img => img.mlAnalysis)
    
    console.log('🫁 ML Analyses:', mlAnalyses.length)
    
    if (mlAnalyses.length === 0) {
      console.log('❌ No ML analyses found')
      return null
    }

    // Aggregate pneumonia analysis results
    const pneumoniaDetections = mlAnalyses.filter(a => 
      a.prediction === 'Pneumonia' || 
      a.prediction === 'pneumonia' ||
      a.metadata?.prediction?.class === 'Pneumonia'
    )
    const avgConfidence = mlAnalyses.reduce((sum, a) => sum + (a.confidence || 0), 0) / mlAnalyses.length

    const result = {
      totalImages: images.length,
      analyzedImages: mlAnalyses.length,
      pneumonia: {
        detected: pneumoniaDetections.length > 0,
        positiveCount: pneumoniaDetections.length,
        status: pneumoniaDetections.length > 0 ? 'Pneumonia Detected' : 'Normal',
        avgConfidence: avgConfidence,
      },
      patientInfo: sample.patientInfo || {},
      analyses: mlAnalyses
    }
    
    console.log('✅ Pneumonia analysis data extracted:', result)
    return result
  }

  // Initialize real-time data from sample
  useEffect(() => {
    if (imageType === 'pneumonia') {
      const pneumoniaData = extractPneumoniaAnalysisData()
      setPneumoniaAnalysisData(pneumoniaData)
      if (pneumoniaData) {
        setIsAnalyzing(true)
        console.log('🫁 Pneumonia analysis data loaded:', pneumoniaData)
      } else {
        setIsAnalyzing(false)
      }
    } else {
      const extracted = extractRealData()
      setRealTimeData(extracted)
      
      if (extracted) {
        setIsAnalyzing(true)
        console.log('🔬 Tissue sample data loaded:', extracted)
      } else {
        setIsAnalyzing(false)
      }
    }
  }, [sample, imageType])

  const getConfidenceData = () => {
    // For pneumonia analysis
    if (imageType === 'pneumonia' && pneumoniaAnalysisData) {
      const baseConfidence = pneumoniaAnalysisData.pneumonia.avgConfidence * 100
      return [
        { 
          category: 'Pneumonia Detection', 
          confidence: Math.min(98, baseConfidence), 
          color: pneumoniaAnalysisData.pneumonia.detected ? '#ef4444' : '#22c55e'
        },
        { 
          category: 'DenseNet121 Confidence', 
          confidence: Math.min(96, baseConfidence + 2), 
          color: '#3b82f6' 
        },
        { 
          category: 'EfficientNet-B0 Confidence', 
          confidence: Math.min(95, baseConfidence + 1), 
          color: '#8b5cf6' 
        },
        { 
          category: 'Ensemble Score', 
          confidence: Math.min(97, baseConfidence + 3), 
          color: '#f59e0b' 
        }
      ]
    }
    
    // For tissue analysis
    if (!realTimeData) {
      return [
        { category: 'Cell Detection', confidence: 96.8, color: '#22c55e' },
        { category: 'Morphology Analysis', confidence: 94.2, color: '#3b82f6' },
        { category: 'Anomaly Detection', confidence: 87.5, color: '#f59e0b' },
        { category: 'Classification', confidence: 91.3, color: '#8b5cf6' }
      ]
    }

    // For tissue analysis - show what EfficientNetB3 model provides
    const tumorConfidence = realTimeData.averageConfidence * 100
    const riskLevel = realTimeData.highRiskCount > 0 ? 'high' : 
                      realTimeData.avgTumorProbability > 50 ? 'medium' : 'low'
    
    return [
      { 
        category: 'Tumor Detection', 
        confidence: tumorConfidence, 
        color: realTimeData.tumorDetected ? '#ef4444' : '#22c55e',
        description: realTimeData.tumorDetected ? 'Tumor detected' : 'No tumor detected'
      },
      { 
        category: 'EfficientNetB3 Confidence', 
        confidence: tumorConfidence, 
        color: tumorConfidence > 80 ? '#22c55e' : tumorConfidence > 60 ? '#f59e0b' : '#ef4444',
        description: 'Brain tumor detection model confidence'
      },
      { 
        category: 'Tumor Probability', 
        confidence: realTimeData.avgTumorProbability, 
        color: realTimeData.avgTumorProbability > 50 ? '#ef4444' : '#22c55e',
        description: `${realTimeData.avgTumorProbability.toFixed(1)}% probability`
      },
      { 
        category: 'Risk Assessment', 
        confidence: riskLevel === 'high' ? 90 : riskLevel === 'medium' ? 60 : 30, 
        color: riskLevel === 'high' ? '#ef4444' : riskLevel === 'medium' ? '#f59e0b' : '#22c55e',
        description: `${riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk`
      }
    ]
  }

  const analysisStages = [
    { id: 'preprocessing', label: 'Image Preprocessing', duration: 15 },
    { id: 'tiling', label: 'WSI Tiling', duration: 25 },
    { id: 'feature_extraction', label: 'Feature Extraction', duration: 30 },
    { id: 'classification', label: 'AI Classification', duration: 20 },
    { id: 'postprocessing', label: 'Result Processing', duration: 10 }
  ]

  const confidenceData = getConfidenceData()

  useEffect(() => {
    if (isAnalyzing) {
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval)
            setIsAnalyzing(false)
            
            // Toast notification for completion
            const analysisType = imageType === 'pneumonia' ? 'Pneumonia analysis' : 'Brain tumor analysis'
            toast.success(`${analysisType} completed!`)
            return 100
          }
          
          // Update current stage based on progress
          const currentProgress = Math.min(100, prev + ((realTimeData || pneumoniaAnalysisData) ? 3 : 2))
          if (currentProgress < 15) setCurrentStage('preprocessing')
          else if (currentProgress < 40) setCurrentStage('tiling')
          else if (currentProgress < 70) setCurrentStage('feature_extraction')
          else if (currentProgress < 90) setCurrentStage('classification')
          else setCurrentStage('postprocessing')
          
          return currentProgress
        })
      }, (realTimeData || pneumoniaAnalysisData) ? 80 : 100)

      return () => clearInterval(interval)
    }
  }, [isAnalyzing, realTimeData, pneumoniaAnalysisData, imageType])

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200'
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low': return 'text-green-600 bg-green-50 border-green-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="grid gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Analysis Dashboard</h1>
          <p className="text-muted-foreground">
            Step 3: AI-powered pathology analysis and prediction
            {(realTimeData?.patientInfo?.name || pneumoniaAnalysisData?.patientInfo?.name) && (
              <span className="ml-2 text-primary">• Patient: {realTimeData?.patientInfo?.name || pneumoniaAnalysisData?.patientInfo?.name}</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(realTimeData || pneumoniaAnalysisData) && (
            <Badge variant="secondary" className="text-sm">
              <Target className="h-4 w-4 mr-2" />
              {(realTimeData?.analyzedImages || pneumoniaAnalysisData?.analyzedImages)}/{(realTimeData?.totalImages || pneumoniaAnalysisData?.totalImages)} Images
            </Badge>
          )}
          <Badge variant="outline" className="text-sm">
            <Brain className="h-4 w-4 mr-2" />
            {isAnalyzing ? 'Processing...' : 'Analysis Complete'}
          </Badge>
        </div>
      </div>

      {/* Pneumonia Analysis Summary Cards - Only show for pneumonia samples */}
      {imageType === 'pneumonia' && pneumoniaAnalysisData && !isAnalyzing && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className={`border-2 ${pneumoniaAnalysisData.pneumonia.detected ? 'border-red-200 bg-red-50/50' : 'border-green-200 bg-green-50/50'}`}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Pneumonia Detection</p>
                  <p className={`text-2xl font-bold ${pneumoniaAnalysisData.pneumonia.detected ? 'text-red-600' : 'text-green-600'}`}>
                    {pneumoniaAnalysisData.pneumonia.detected ? 'PNEUMONIA' : 'NORMAL'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {pneumoniaAnalysisData.pneumonia.positiveCount} of {pneumoniaAnalysisData.analyzedImages} images
                  </p>
                </div>
                {pneumoniaAnalysisData.pneumonia.detected ? 
                  <AlertTriangle className="h-8 w-8 text-red-600" /> : 
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                }
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Model Confidence</p>
                  <p className="text-2xl font-bold text-blue-600">{(pneumoniaAnalysisData.pneumonia.avgConfidence * 100).toFixed(1)}%</p>
                  <p className="text-xs text-muted-foreground">ensemble average</p>
                </div>
                <TrendingUp className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Images Analyzed</p>
                  <p className="text-2xl font-bold text-purple-600">{pneumoniaAnalysisData.analyzedImages}</p>
                  <p className="text-xs text-muted-foreground">chest X-rays</p>
                </div>
                <Target className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Model</p>
                  <p className="text-2xl font-bold text-orange-600">Ensemble</p>
                  <p className="text-xs text-muted-foreground">DenseNet + EfficientNet</p>
                </div>
                <Activity className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Tissue Analysis Summary Cards */}
      {imageType === 'tissue' && realTimeData && !isAnalyzing && (() => {
        // Threshold: if tumor probability < 54%, consider it NEGATIVE
        const tumorThreshold = 54
        const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
        
        return (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            
            <Card className={`border-2 ${isTumorPositive ? 'border-red-200 bg-red-50/50' : 'border-green-200 bg-green-50/50'}`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Tumor Detection</p>
            <p className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
              {isTumorPositive ? 'POSITIVE' : 'NEGATIVE'}
            </p>
            <p className="text-xs text-muted-foreground">
              {realTimeData.tumorCount} of {realTimeData.analyzedImages} images
            </p>
              </div>
              {isTumorPositive ? (
                <AlertTriangle className="h-8 w-8 text-red-600" />
              ) : (
                <CheckCircle2 className="h-8 w-8 text-green-600" />
              )}
            </div>
          </CardContent>
            </Card>

            <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Overall Confidence</p>
            <p className="text-2xl font-bold">{(realTimeData.averageConfidence * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
            </Card>

            <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Tumor Probability</p>
            <p className="text-2xl font-bold text-orange-600">
              {realTimeData.avgTumorProbability.toFixed(1)}%
            </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
            </Card>

            <Card className={`border-2 ${
              realTimeData.highRiskCount > 0 ? 'border-red-200 bg-red-50/50' : 
              realTimeData.avgTumorProbability > 50 ? 'border-yellow-200 bg-yellow-50/50' : 
              'border-green-200 bg-green-50/50'
            }`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Risk Level</p>
            <p className={`text-2xl font-bold ${
              realTimeData.highRiskCount > 0 ? 'text-red-600' : 
              realTimeData.avgTumorProbability > 50 ? 'text-yellow-600' : 
              'text-green-600'
            }`}>
              {realTimeData.highRiskCount > 0 ? 'HIGH' : 
               realTimeData.avgTumorProbability > 50 ? 'MEDIUM' : 'LOW'}
            </p>
              </div>
              <AlertTriangle className={`h-8 w-8 ${
                realTimeData.highRiskCount > 0 ? 'text-red-600' : 
                realTimeData.avgTumorProbability > 50 ? 'text-yellow-600' : 
                'text-green-600'
              }`} />
            </div>
          </CardContent>
            </Card>
          </div>
        )
      })()}

      {/* Tumor Type Classification - Only for Tissue with tumor data */}
      {imageType === 'tissue' && realTimeData && !isAnalyzing && (() => {
        // Extract tumor classification from ML analyses
        const analyses = realTimeData.analyses || []
        const tumorTypes = analyses
          .map(a => ({
            tumorType: a.tumorType || a.metadata?.prediction?.predicted_class || null,
            tumorTypeInfo: a.tumorTypeInfo || null,
            classProbabilities: a.classProbabilities || a.metadata?.prediction?.probabilities || {},
            confidence: a.confidence || 0
          }))
          .filter(a => a.tumorType)

        if (tumorTypes.length === 0) return null

        // Get the primary (highest confidence) tumor type
        const primary = tumorTypes.sort((a, b) => b.confidence - a.confidence)[0]
        const tumorClass = primary.tumorType?.toLowerCase()
        const isNoTumor = tumorClass === 'notumor' || tumorClass === 'no tumor'

        // Map for display
        const classDisplayInfo = {
          glioma: { label: 'Glioma', color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200', barColor: 'bg-red-500', icon: '🔴', description: 'A tumor arising from glial cells in the brain or spine. Gliomas are the most common type of primary brain tumor.' },
          meningioma: { label: 'Meningioma', color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-200', barColor: 'bg-amber-500', icon: '🟠', description: 'A tumor arising from the meninges, the membranes surrounding the brain and spinal cord. Usually slow-growing and often benign.' },
          pituitary: { label: 'Pituitary Tumor', color: 'text-purple-600', bg: 'bg-purple-50', border: 'border-purple-200', barColor: 'bg-purple-500', icon: '🟣', description: 'An abnormal growth in the pituitary gland at the base of the brain. Can affect hormone production and vision.' },
          notumor: { label: 'No Tumor', color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200', barColor: 'bg-green-500', icon: '🟢', description: 'No tumor detected in the brain MRI scan. The scan appears normal.' }
        }

        const displayInfo = classDisplayInfo[tumorClass] || classDisplayInfo.notumor
        const probabilities = primary.classProbabilities || {}

        // Ensure probabilities are in percentage format
        const normalizedProbs = {}
        for (const [key, val] of Object.entries(probabilities)) {
          normalizedProbs[key] = val > 1 ? val : val * 100
        }

        return (
          <Card className={`border-2 ${displayInfo.border} ${displayInfo.bg}/30`}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Tumor Type Classification
                <Badge variant="outline" className="ml-auto text-xs">
                  4-Class EfficientNetB3
                </Badge>
              </CardTitle>
              <CardDescription>
                Detailed classification across all 4 tumor categories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-5">
                {/* Primary Detection Result */}
                <div className={`flex items-center gap-4 p-4 rounded-xl border-2 ${displayInfo.border} ${displayInfo.bg}`}>
                  <div className="text-4xl">{displayInfo.icon}</div>
                  <div className="flex-1">
                    <div className={`text-2xl font-bold ${displayInfo.color}`}>
                      {displayInfo.label}
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">{displayInfo.description}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-3xl font-bold ${displayInfo.color}`}>
                      {(normalizedProbs[tumorClass] || primary.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">confidence</div>
                  </div>
                </div>

                {/* All 4 Classes Probability Bars */}
                <div className="space-y-3">
                  <h4 className="font-semibold text-sm flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Class Probability Distribution
                  </h4>
                  {['glioma', 'meningioma', 'pituitary', 'notumor'].map(cls => {
                    const info = classDisplayInfo[cls]
                    const prob = normalizedProbs[cls] || 0
                    const isDetected = cls === tumorClass
                    return (
                      <div key={cls} className={`flex items-center gap-3 p-2.5 rounded-lg transition-all ${isDetected ? `${info.bg} ${info.border} border-2 shadow-sm` : 'bg-gray-50 border border-transparent'}`}>
                        <div className="text-lg w-6 text-center">{info.icon}</div>
                        <div className="w-28 text-sm font-medium">
                          {info.label}
                          {isDetected && <Badge variant="secondary" className="ml-1.5 text-[10px] px-1 py-0">Detected</Badge>}
                        </div>
                        <div className="flex-1">
                          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div 
                              className={`h-3 rounded-full transition-all duration-1000 ${info.barColor}`}
                              style={{ width: `${Math.min(prob, 100)}%` }}
                            />
                          </div>
                        </div>
                        <div className={`text-sm font-bold w-16 text-right ${isDetected ? info.color : 'text-gray-500'}`}>
                          {prob.toFixed(1)}%
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Severity Note */}
                {!isNoTumor && (
                  <Alert className="border-amber-200 bg-amber-50/50">
                    <AlertTriangle className="h-4 w-4 text-amber-600" />
                    <AlertTitle className="text-amber-800">Clinical Note</AlertTitle>
                    <AlertDescription className="text-amber-700 text-sm">
                      {tumorClass === 'glioma' && 'Gliomas require immediate specialist consultation. They can range from low-grade (slow-growing) to high-grade (aggressive). Further imaging and biopsy are recommended.'}
                      {tumorClass === 'meningioma' && 'Meningiomas are usually benign and slow-growing. Regular monitoring with follow-up MRI scans is recommended. Surgical intervention may be needed if symptoms develop.'}
                      {tumorClass === 'pituitary' && 'Pituitary tumors may affect hormone levels. Endocrinological evaluation and hormone panel tests are recommended alongside follow-up imaging.'}
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        )
      })()}

        {/* Tumor Percentage Detailed Analysis - Only for Tissue */}
      {imageType === 'tissue' && realTimeData && !isAnalyzing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Tumor Distribution Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Tumor Distribution Analysis
              </CardTitle>
              <CardDescription>
                Percentage breakdown of tumor vs normal tissue detected
              </CardDescription>
            </CardHeader>
            <CardContent>
              {(() => {
                // Threshold: if tumor probability < 54%, consider it Normal
                const tumorThreshold = 54
                const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
                const displayTumorPercent = isTumorPositive ? realTimeData.avgTumorProbability : 0
                const displayNormalPercent = isTumorPositive ? (100 - realTimeData.avgTumorProbability) : 100
                const tumorImageCount = isTumorPositive ? realTimeData.tumorCount : 0
                const normalImageCount = isTumorPositive ? (realTimeData.analyzedImages - realTimeData.tumorCount) : realTimeData.analyzedImages
                
                return (
              <div className="space-y-4">
                {/* Large Percentage Display */}
                <div className={`text-center p-6 rounded-lg border-2 ${
                  isTumorPositive 
                    ? 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200' 
                    : 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
                }`}>
                  <div className={`text-6xl font-bold mb-2 ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {realTimeData.avgTumorProbability.toFixed(1)}%
                  </div>
                  <div className={`text-lg font-medium ${isTumorPositive ? 'text-red-700' : 'text-green-700'}`}>
                    {isTumorPositive ? 'Tumor Detected in Samples' : 'Normal Tissue (Below Threshold)'}
                  </div>
                  <div className={`text-sm mt-2 ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {isTumorPositive ? `${realTimeData.tumorCount} out of ${realTimeData.analyzedImages} images` : 'All samples appear normal'}
                  </div>
                </div>

                {/* Breakdown */}
                <div className="space-y-3">
                  <div className={`flex items-center justify-between p-3 rounded-lg ${isTumorPositive ? 'bg-red-100' : 'bg-gray-100'}`}>
                    <div className="flex items-center gap-3">
                      <div className={`w-4 h-4 rounded-full ${isTumorPositive ? 'bg-red-600' : 'bg-gray-400'}`}></div>
                      <span className="font-medium">Tumor Detected</span>
                    </div>
                    <div className={`font-bold ${isTumorPositive ? 'text-red-600' : 'text-gray-500'}`}>
                      {displayTumorPercent.toFixed(1)}% ({tumorImageCount} images)
                    </div>
                  </div>
                  
                  <div className={`flex items-center justify-between p-3 rounded-lg ${!isTumorPositive ? 'bg-green-100' : 'bg-green-50'}`}>
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-green-600"></div>
                      <span className="font-medium">Normal Tissue</span>
                    </div>
                    <div className="font-bold text-green-600">
                      {displayNormalPercent.toFixed(1)}% ({normalImageCount} images)
                    </div>
                  </div>
                </div>
              </div>
                )
              })()}
            </CardContent>
          </Card>

          {/* Tumor Probability Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Tumor Probability Analysis
              </CardTitle>
              <CardDescription>
                Detailed probability scores from AI model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Probability Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <div className="text-sm font-medium text-orange-700">Average Probability</div>
                    <div className="text-2xl font-bold text-orange-600">
                      {realTimeData.avgTumorProbability.toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                    <div className="text-sm font-medium text-red-700">Maximum Probability</div>
                    <div className="text-2xl font-bold text-red-600">
                      {realTimeData.maxTumorProbability.toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Per-Image Tumor Probabilities */}
                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">Per-Image Tumor Probabilities</h4>
                  {realTimeData.analyses.map((analysis, index) => {
                    // Try multiple possible locations for tumor probability
                    let rawProb = 
                      analysis.metadata?.probabilities?.tumor ||
                      analysis.metadata?.probabilities?.Tumor ||
                      analysis.probabilities?.tumor ||
                      (analysis.metadata?.is_tumor ? analysis.metadata?.confidence || analysis.confidence : null) ||
                      (analysis.prediction === 'malignant' || analysis.prediction === 'Tumor' ? analysis.confidence : null) ||
                      0
                    // Convert to percentage if it's a decimal
                    const tumorProb = rawProb > 1 ? rawProb : rawProb * 100
                    return (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span className="text-sm font-medium">Image {index + 1}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              tumorProb > 50 ? 'bg-red-500' : 'bg-yellow-500'
                            }`}
                            style={{ width: `${Math.min(tumorProb, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-bold">
                          {tumorProb.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    )
                  })}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Pneumonia Analysis Detailed Card */}
      {imageType === 'pneumonia' && pneumoniaAnalysisData && !isAnalyzing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Pneumonia Detection Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Pneumonia Detection Analysis
              </CardTitle>
              <CardDescription>
                DenseNet121 + EfficientNet-B0 ensemble chest X-ray analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className={`text-center p-6 rounded-lg border-2 ${
                  pneumoniaAnalysisData.pneumonia.detected 
                    ? 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200' 
                    : 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
                }`}>
                  <div className={`text-4xl font-bold mb-2 ${
                    pneumoniaAnalysisData.pneumonia.detected ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {pneumoniaAnalysisData.pneumonia.status}
                  </div>
                  <div className={`text-lg font-medium ${
                    pneumoniaAnalysisData.pneumonia.detected ? 'text-red-700' : 'text-green-700'
                  }`}>
                    {pneumoniaAnalysisData.pneumonia.detected ? 'Pneumonia Indicators Found' : 'No Pneumonia Detected'}
                  </div>
                  <div className="text-sm mt-2 text-muted-foreground">
                    Confidence: {(pneumoniaAnalysisData.pneumonia.avgConfidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg border">
                    <div className="text-sm font-medium text-muted-foreground">Positive Images</div>
                    <div className="text-2xl font-bold text-red-600">
                      {pneumoniaAnalysisData.pneumonia.positiveCount}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg border">
                    <div className="text-sm font-medium text-muted-foreground">Normal Images</div>
                    <div className="text-2xl font-bold text-green-600">
                      {pneumoniaAnalysisData.analyzedImages - pneumoniaAnalysisData.pneumonia.positiveCount}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model Ensemble Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Ensemble Model Breakdown
              </CardTitle>
              <CardDescription>
                Dual-model ensemble for robust pneumonia classification
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200">
                  <div className="text-4xl font-bold text-blue-600 mb-1">
                    {(pneumoniaAnalysisData.pneumonia.avgConfidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm font-medium text-blue-700">Ensemble Confidence</div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-blue-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-blue-600"></div>
                      <span className="font-medium">DenseNet121</span>
                    </div>
                    <div className="font-bold text-blue-600">Primary Model</div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-purple-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-purple-600"></div>
                      <span className="font-medium">EfficientNet-B0</span>
                    </div>
                    <div className="font-bold text-purple-600">Secondary Model</div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-green-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-green-600"></div>
                      <span className="font-medium">Weighted Average</span>
                    </div>
                    <div className="font-bold text-green-600">Ensemble Strategy</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Analysis Progress */}
      {isAnalyzing && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 animate-pulse" />
              Analysis in Progress
            </CardTitle>
            <CardDescription>
              AI model is processing your samples using multi-scale attention
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Overall Progress</span>
                <span className="text-sm text-muted-foreground">{analysisProgress.toFixed(0)}%</span>
              </div>
              
              <div className="w-full bg-muted rounded-full h-3">
                <div 
                  className="bg-primary h-3 rounded-full transition-all duration-300"
                  style={{ width: `${analysisProgress}%` }}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                {analysisStages.map((stage) => (
                  <div 
                    key={stage.id}
                    className={`p-3 rounded-lg text-center ${
                      currentStage === stage.id 
                        ? 'bg-primary text-primary-foreground' 
                        : analysisProgress > analysisStages.findIndex(s => s.id === stage.id) * 20
                        ? 'bg-green-100 text-green-800'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    <div className="text-xs font-medium">{stage.label}</div>
                    {currentStage === stage.id && (
                      <div className="text-xs mt-1 flex items-center justify-center gap-1">
                        <Clock className="h-3 w-3" />
                        ~{stage.duration}s
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="pneumonia-analysis" className={imageType === 'pneumonia' ? 'bg-red-100' : ''}>
            🫁 Pneumonia Detection
          </TabsTrigger>
          <TabsTrigger value="tissue-analysis" className={imageType === 'tissue' ? 'bg-blue-100' : ''}>
            🧠 Brain Tumor Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pneumonia-analysis" className="space-y-6">
          {imageType !== 'pneumonia' ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-semibold mb-2">Pneumonia Analysis Not Applicable</h3>
                <p className="text-muted-foreground">
                  This analysis is only available for chest X-ray images. 
                  The current sample is a tissue image for brain tumor detection.
                </p>
              </CardContent>
            </Card>
          ) : !pneumoniaAnalysisData ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-600" />
                <h3 className="text-lg font-semibold mb-2">No Pneumonia Analysis Data</h3>
                <p className="text-muted-foreground">
                  Chest X-ray analysis results will appear here after uploading and analyzing chest X-ray images.
                  Select "Chest X-ray (Pneumonia Detection)" in Step 1 to run pneumonia detection.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {/* Pneumonia Detection Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Pneumonia Detection Results
                  </CardTitle>
                  <CardDescription>
                    DenseNet121 + EfficientNet-B0 ensemble chest X-ray analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-4">
                      <Skeleton className="h-32 w-full" />
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Alert className={pneumoniaAnalysisData.pneumonia.detected ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                        {pneumoniaAnalysisData.pneumonia.detected ? (
                          <>
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                            <AlertTitle className="text-red-600">Pneumonia Detected!</AlertTitle>
                            <AlertDescription>
                              {pneumoniaAnalysisData.pneumonia.positiveCount} image(s) show signs of pneumonia. 
                              Medical review recommended.
                            </AlertDescription>
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="h-4 w-4 text-green-600" />
                            <AlertTitle className="text-green-600">No Pneumonia Detected</AlertTitle>
                            <AlertDescription>
                              All {pneumoniaAnalysisData.analyzedImages} chest X-ray images appear normal.
                            </AlertDescription>
                          </>
                        )}
                      </Alert>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className={`text-2xl font-bold ${pneumoniaAnalysisData.pneumonia.detected ? 'text-red-600' : 'text-green-600'}`}>
                              {pneumoniaAnalysisData.pneumonia.status}
                            </div>
                            <div className="text-sm text-muted-foreground">Detection Status</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className="text-2xl font-bold">{(pneumoniaAnalysisData.pneumonia.avgConfidence * 100).toFixed(1)}%</div>
                            <div className="text-sm text-muted-foreground">Model Confidence</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className="text-2xl font-bold">{pneumoniaAnalysisData.analyzedImages}</div>
                            <div className="text-sm text-muted-foreground">Images Analyzed</div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="tissue-analysis" className="space-y-6">
          {imageType !== 'tissue' ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-semibold mb-2">Brain Tumor Analysis Not Applicable</h3>
                <p className="text-muted-foreground">
                  This analysis is only available for tissue/histopathology images. 
                  The current sample is a chest X-ray for pneumonia detection.
                </p>
              </CardContent>
            </Card>
          ) : !realTimeData ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-600" />
                <h3 className="text-lg font-semibold mb-2">No Tissue Analysis Data</h3>
                <p className="text-muted-foreground">
                  Tissue biopsy analysis results will appear here after analyzing histopathology images.
                </p>
              </CardContent>
            </Card>
          ) : (
          <Card>
            <CardHeader>
              <CardTitle>Tissue Biopsy Analysis</CardTitle>
              <CardDescription>
                AI-powered cancer detection and morphological analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isAnalyzing ? (
                <div className="space-y-4">
                  <Skeleton className="h-32 w-full" />
                  <div className="grid grid-cols-3 gap-4">
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                </div>
              ) : (() => {
                // Threshold: if tumor probability < 54%, consider it Normal
                const tumorThreshold = 54
                const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
                
                return (
                <div className="space-y-4">
                  <Alert className={isTumorPositive ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                    {isTumorPositive ? (
                      <>
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                        <AlertTitle className="text-red-600">Abnormality Detected</AlertTitle>
                        <AlertDescription>
                          {realTimeData.tumorCount} suspicious region(s) detected. Further review recommended.
                        </AlertDescription>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                        <AlertTitle className="text-green-600">Analysis Complete - Normal</AlertTitle>
                        <AlertDescription>
                          No malignant cells detected. Analysis indicates normal tissue.
                        </AlertDescription>
                      </>
                    )}
                  </Alert>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                          {isTumorPositive ? 'Abnormal' : 'Normal'}
                        </div>
                        <div className="text-sm text-muted-foreground">Overall Assessment</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold">{(realTimeData.averageConfidence * 100).toFixed(1)}%</div>
                        <div className="text-sm text-muted-foreground">Confidence Score</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                          {isTumorPositive ? realTimeData.tumorCount : 0}
                        </div>
                        <div className="text-sm text-muted-foreground">Suspicious Regions</div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
                )
              })()}
            </CardContent>
          </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Action Button */}
      <div className="flex justify-end">
        <Button 
          onClick={() => {
            toast.success("Proceeding to results review...")
            setTimeout(() => onNext(), 1500)
          }}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? 'Analysis in Progress...' : 'Proceed to Review'}
        </Button>
      </div>
    </div>
  )
}