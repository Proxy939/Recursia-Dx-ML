import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { toast } from 'sonner'
import { 
  CheckCircle2, 
  XCircle, 
  AlertTriangle, 
  FileCheck,
  ArrowRight,
  Microscope,
  Stethoscope,
  Activity,
  MessageSquare,
  Shield
} from 'lucide-react'

export function ResultsReview({ onNext, sample }) {
  const [approval, setApproval] = useState(null) // 'approved' | 'reanalysis' | null
  const [notes, setNotes] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Determine sample type
  const sampleType = sample?.sampleType || 'Brain MRI'
  const isPneumonia = sampleType.toLowerCase().includes('chest') || sampleType.toLowerCase().includes('pneumonia') || sample?.imageType === 'pneumonia'

  // Extract analysis results
  const getAnalysisResults = () => {
    if (!sample?.images || sample.images.length === 0) {
      return null
    }

    const firstImage = sample.images[0]
    if (!firstImage?.mlAnalysis) {
      return null
    }

    const analysis = firstImage.mlAnalysis

    if (isPneumonia) {
      // Pneumonia analysis — uses different data structure
      const confidence = analysis.confidence || 0
      const confidencePct = confidence <= 1 ? confidence * 100 : confidence
      const detected = analysis.isPneumonia || analysis.prediction === 'malignant' || analysis.prediction === 'Pneumonia'

      return {
        prediction: detected ? 'Pneumonia Detected' : 'Normal',
        confidence: confidencePct.toFixed(1),
        riskLevel: detected ? 'High' : 'Low',
        probabilities: analysis.classProbabilities || {},
        tumorProbability: confidencePct / 100,
        isPositive: detected,
        severity: analysis.severity || 'Unknown',
        detectedFeatures: []
      }
    }
    
    // Brain tumor analysis - use class probabilities
    const probs = analysis.classProbabilities || analysis.metadata?.prediction?.probabilities || analysis.metadata?.probabilities || {}
    let notumorProb = probs.notumor ?? probs.Notumor ?? probs['no tumor'] ?? probs['No Tumor'] ?? null
    let tumorProb
    if (notumorProb !== null) {
      if (notumorProb <= 1) notumorProb = notumorProb * 100
      tumorProb = 100 - notumorProb
    } else {
      tumorProb = analysis.prediction === 'malignant' ? (analysis.confidence <= 1 ? analysis.confidence * 100 : analysis.confidence) : 0
    }
    const isPositive = tumorProb >= 54
    const riskLevel = tumorProb >= 70 ? 'High' : tumorProb >= 50 ? 'Medium' : 'Low'
    
    // Get detected tumor type
    const tumorType = analysis.tumorType || analysis.metadata?.prediction?.predicted_class || null
    const predictedClass = tumorType?.toLowerCase()
    const tumorLabel = predictedClass === 'glioma' ? 'Glioma Detected' :
                       predictedClass === 'meningioma' ? 'Meningioma Detected' :
                       predictedClass === 'pituitary' ? 'Pituitary Tumor Detected' :
                       predictedClass === 'notumor' ? 'No Tumor Detected' :
                       isPositive ? 'Tumor Detected' : 'No Tumor Detected'
    
    return {
      prediction: tumorLabel,
      confidence: (analysis.confidence <= 1 ? analysis.confidence * 100 : analysis.confidence).toFixed(1),
      riskLevel,
      probabilities: probs,
      tumorProbability: tumorProb / 100,
      isPositive,
      detectedFeatures: analysis.detected_features || analysis.metadata?.detected_features || []
    }
  }

  const analysisResults = getAnalysisResults()

  // Handle submission
  const handleSubmit = () => {
    if (!approval) {
      toast.error('Please select an approval option')
      return
    }

    setIsSubmitting(true)

    setTimeout(() => {
      setIsSubmitting(false)
      if (approval === 'approved') {
        toast.success('Results verified! Proceeding to report generation...')
      } else {
        toast.info('Sample flagged for re-analysis')
      }
      setTimeout(() => onNext(), 1000)
    }, 1500)
  }

  // If no sample or no analysis
  if (!sample || !analysisResults) {
    return (
      <div className="grid gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Results Review</h1>
            <p className="text-muted-foreground">
              Step 4: Verify AI analysis before report generation
            </p>
          </div>
        </div>

        <Card className="border-amber-200 bg-amber-50">
          <CardContent className="p-12 text-center">
            <AlertTriangle className="h-16 w-16 mx-auto text-amber-500 mb-4" />
            <h2 className="text-xl font-semibold mb-2">No Result Available</h2>
            <p className="text-muted-foreground max-w-md mx-auto mb-6">
              Please complete Step 1 (Upload) and Step 3 (AI Analysis) before reviewing results.
            </p>
            <Alert className="max-w-lg mx-auto text-left">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Required Steps</AlertTitle>
              <AlertDescription>
                1. Upload sample images → 2. Enter patient info → 3. Run AI Analysis → 4. Review here
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Use the extracted tumor probability
  const confidenceThreshold = 54
  const tumorProb = analysisResults.tumorProbability * 100
  const isPositive = tumorProb >= confidenceThreshold || analysisResults.isPositive

  return (
    <div className="grid gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Results Review</h1>
          <p className="text-muted-foreground">
            Step 4: Verify AI analysis before report generation
          </p>
        </div>
        <Badge variant="outline" className="text-sm">
          <FileCheck className="h-4 w-4 mr-2" />
          Quality Control
        </Badge>
      </div>

      {/* Sample Info Bar */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              {isPneumonia ? (
                <Stethoscope className="h-8 w-8 text-orange-500" />
              ) : (
                <Microscope className="h-8 w-8 text-blue-500" />
              )}
              <div>
                <div className="font-semibold">{sample.sampleId || 'Sample'}</div>
                <div className="text-sm text-muted-foreground">
                  {sampleType} • {sample.images?.length || 0} image(s) analyzed
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-muted-foreground">Patient</div>
              <div className="font-medium">
                {sample.patientInfo?.name || 'Unknown'}, {sample.patientInfo?.age || '?'} years
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Analysis Result */}
      <Card className={`border-2 ${isPositive ? 'border-red-300 bg-red-50/50' : 'border-green-300 bg-green-50/50'}`}>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            {isPneumonia ? 'Chest X-ray Analysis Result' : 'Brain Tumor Analysis Result'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Primary Result */}
            <div className="text-center p-6 bg-white rounded-lg border">
              <div className={`text-4xl font-bold mb-2 ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                {isPositive ? (isPneumonia ? 'PNEUMONIA DETECTED' : 'TUMOR DETECTED') : (isPneumonia ? 'NORMAL' : 'NO TUMOR')}
              </div>
              <div className="text-lg text-muted-foreground">
                {isPneumonia ? 'Pneumonia Detection' : 'Brain Tumor Detection'}
              </div>
            </div>

            {/* Metrics */}
            <div className="space-y-4">
              <div className="p-4 bg-white rounded-lg border">
                <div className="text-sm text-muted-foreground">Confidence Score</div>
                <div className="text-2xl font-bold">{analysisResults.confidence}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div 
                    className={`h-2 rounded-full ${parseFloat(analysisResults.confidence) >= 80 ? 'bg-green-500' : parseFloat(analysisResults.confidence) >= 60 ? 'bg-yellow-500' : 'bg-red-500'}`}
                    style={{ width: `${analysisResults.confidence}%` }}
                  />
                </div>
              </div>

              <div className="p-4 bg-white rounded-lg border">
                <div className="text-sm text-muted-foreground">Risk Assessment</div>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant={analysisResults.riskLevel?.includes('High') ? 'destructive' : 
                                 analysisResults.riskLevel?.includes('Moderate') ? 'secondary' : 'outline'}>
                    {analysisResults.riskLevel || 'Low Risk'}
                  </Badge>
                </div>
              </div>

              <div className="p-4 bg-white rounded-lg border">
                <div className="text-sm text-muted-foreground">
                  {isPneumonia ? 'Detection Probability' : 'Tumor Probability'}
                </div>
                <div className="text-xl font-semibold">
                  {tumorProb.toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Approval Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Reviewer Verification
          </CardTitle>
          <CardDescription>
            Please review the AI analysis and select an action
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Approval Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div 
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                approval === 'approved' 
                  ? 'border-green-500 bg-green-50' 
                  : 'border-gray-200 hover:border-green-300 hover:bg-green-50/50'
              }`}
              onClick={() => setApproval('approved')}
            >
              <div className="flex items-center gap-3">
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  approval === 'approved' ? 'border-green-500 bg-green-500' : 'border-gray-300'
                }`}>
                  {approval === 'approved' && <div className="w-2 h-2 bg-white rounded-full" />}
                </div>
                <div>
                  <div className="font-semibold text-green-700">✓ Approve Results</div>
                  <div className="text-sm text-muted-foreground">
                    I confirm this analysis is accurate
                  </div>
                </div>
              </div>
            </div>

            <div 
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                approval === 'reanalysis' 
                  ? 'border-amber-500 bg-amber-50' 
                  : 'border-gray-200 hover:border-amber-300 hover:bg-amber-50/50'
              }`}
              onClick={() => setApproval('reanalysis')}
            >
              <div className="flex items-center gap-3">
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                  approval === 'reanalysis' ? 'border-amber-500 bg-amber-500' : 'border-gray-300'
                }`}>
                  {approval === 'reanalysis' && <div className="w-2 h-2 bg-white rounded-full" />}
                </div>
                <div>
                  <div className="font-semibold text-amber-700">⚠ Request Re-analysis</div>
                  <div className="text-sm text-muted-foreground">
                    Sample needs additional review
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Optional Notes */}
          <div>
            <label className="text-sm font-medium flex items-center gap-2 mb-2">
              <MessageSquare className="h-4 w-4" />
              Additional Notes (Optional)
            </label>
            <textarea
              className="w-full p-3 border rounded-lg resize-none"
              rows={3}
              placeholder="Add any observations or comments..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Submit Button */}
      <div className="flex justify-end">
        <Button 
          size="lg"
          disabled={!approval || isSubmitting}
          onClick={handleSubmit}
          className="min-w-[200px]"
        >
          {isSubmitting ? (
            'Submitting...'
          ) : (
            <>
              Submit & Continue
              <ArrowRight className="h-4 w-4 ml-2" />
            </>
          )}
        </Button>
      </div>
    </div>
  )
}