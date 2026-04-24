import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { toast } from 'sonner'
import { Upload, User, FileImage, AlertCircle, CheckCircle2, Camera, Stethoscope, Brain, Loader2, Play, FlaskConical, Microscope } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api'

export function SampleUpload({ onNext, onSampleCreated }) {
  const [patientData, setPatientData] = useState({
    patientInfo: {
      patientId: '',
      name: '',
      age: '',
      gender: '',
      contactNumber: '',
      referringPhysician: ''
    },
    imageType: '', // must be explicitly chosen by user: 'tissue' or 'pneumonia'
    clinicalInfo: {
      clinicalDiagnosis: '',
      symptoms: [],
      duration: '',
      urgency: 'Routine',
      medicalHistory: ''
    }
  })
  
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [mlResults, setMlResults] = useState(null)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [showDemoDialog, setShowDemoDialog] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [demoCategory, setDemoCategory] = useState(null) // 'pneumonia' | 'brain-tumor' | null

  // Dynamic symptoms based on selected analysis type
  const brainTumorSymptoms = [
    'Headaches', 'Seizures', 'Vision Problems', 'Nausea/Vomiting',
    'Memory Loss', 'Confusion', 'Balance Issues', 'Speech Difficulty',
    'Personality Changes', 'Numbness/Tingling', 'Fatigue', 'Dizziness'
  ]

  const pneumoniaSymptoms = [
    'Persistent Cough', 'Chest Pain', 'Shortness of Breath', 'Fever',
    'Chills', 'Fatigue', 'Rapid Breathing', 'Wheezing',
    'Loss of Appetite', 'Body Aches', 'Confusion (elderly)', 'Coughing Blood'
  ]

  const symptoms = patientData.imageType === 'pneumonia' ? pneumoniaSymptoms : brainTumorSymptoms


  const handleInputChange = (section, field, value) => {
    setPatientData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }))
  }

  const handleSymptomChange = (symptom) => {
    setPatientData(prev => ({
      ...prev,
      clinicalInfo: {
        ...prev.clinicalInfo,
        symptoms: prev.clinicalInfo.symptoms.includes(symptom) 
          ? prev.clinicalInfo.symptoms.filter(s => s !== symptom)
          : [...prev.clinicalInfo.symptoms, symptom]
      }
    }))
  }

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files)
    
    // Validate file types (no SVS/NDPI)
    const validFiles = files.filter(file => {
      const ext = file.name.toLowerCase().split('.').pop()
      if (['svs', 'ndpi'].includes(ext)) {
        toast.error(`${file.name}: ${ext.toUpperCase()} format is not supported. Use JPEG, PNG, BMP, or DICOM (.dcm).`)
        return false
      }
      return true
    })
    
    if (validFiles.length === 0) return
    
    // Add files to upload queue
    const newFiles = validFiles.map(file => {
      const isDicom = file.name.toLowerCase().endsWith('.dcm')
      return {
        id: Date.now() + Math.random(),
        file: file,
        name: file.name,
        size: file.size,
        type: file.type,
        isDicom: isDicom,
        uploadedAt: new Date().toISOString(),
        // DICOM files can't be previewed in browser — use null
        preview: isDicom ? null : URL.createObjectURL(file)
      }
    })
    
    setUploadedFiles(prev => [...prev, ...newFiles])
    
    // Auto-detect analysis type from file extension
    const hasDicom = newFiles.some(f => f.isDicom)
    if (hasDicom && !patientData.imageType) {
      setPatientData(prev => ({ ...prev, imageType: 'pneumonia' }))
      toast.info('DICOM file detected — automatically selected Chest X-ray / Pneumonia analysis.')
    }
    
    toast.success(`${validFiles.length} file(s) added`)
  }

  const removeFile = (fileId) => {
    setUploadedFiles(prev => {
      const updated = prev.filter(f => f.id !== fileId)
      // Clean up preview URLs
      const removed = prev.find(f => f.id === fileId)
      if (removed && removed.preview) {
        URL.revokeObjectURL(removed.preview)
      }
      return updated
    })
  }

  // Handle Demo Mode - Process preloaded samples
  const handleDemoSubmit = async (demoType) => {
    setShowDemoDialog(false)
    setDemoCategory(null)
    setIsDemoMode(true)
    setIsAnalyzing(true)
    setAnalysisProgress(0)
    
    // Determine category for patient data
    const isPneumonia = demoType.startsWith('pneumonia')
    const DEMO_LABELS = {
      'pneumonia-positive': 'Pneumonia Positive',
      'pneumonia-negative': 'Normal Chest X-Ray',
      'glioma': 'Glioma Sample',
      'meningioma': 'Meningioma Sample',
      'pituitary': 'Pituitary Tumor Sample',
      'no-tumor': 'Normal Brain MRI',
    }

    // Auto-fill demo patient data
    setPatientData(prev => ({
      ...prev,
      patientInfo: {
        patientId: `DEMO-${Date.now()}`,
        name: `Demo Patient (${DEMO_LABELS[demoType] || demoType})`,
        age: '45',
        gender: 'Female'
      },
      imageType: isPneumonia ? 'pneumonia' : 'tissue'
    }))
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 500)
      
      // Call backend demo endpoint
      const response = await fetch(`${API_BASE_URL}/samples/demo-analysis?type=${demoType}`, {
        method: 'POST'
      })
      
      clearInterval(progressInterval)
      setAnalysisProgress(100)
      
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        const errMsg = typeof errData.error === 'string' ? errData.error : (errData.message || 'Demo analysis failed')
        throw new Error(errMsg)
      }
      
      const result = await response.json()
      console.log('✅ Demo result:', result)
      
      setMlResults(result)
      toast.success(`Demo ${DEMO_LABELS[demoType]} analyzed successfully!`)
      
      if (onSampleCreated) {
        onSampleCreated(result.data.sample)
      }
      
      setTimeout(() => {
        setIsAnalyzing(false)
        setIsDemoMode(false)
        onNext()
      }, 1000)
      
    } catch (error) {
      console.error('Demo error:', error)
      setIsAnalyzing(false)
      setIsDemoMode(false)
      setAnalysisProgress(0)
      toast.error(error.message || 'Failed to process demo sample')
    }
  }

  const handleSubmit = async () => {
    // Validation - auto-fill patient data if not provided
    if (!patientData.patientInfo.patientId || !patientData.patientInfo.name) {
      // Auto-fill instead of blocking
      setPatientData(prev => ({
        ...prev,
        patientInfo: {
          ...prev.patientInfo,
          patientId: prev.patientInfo.patientId || `PT-${Date.now()}`,
          name: prev.patientInfo.name || 'Anonymous Patient'
        }
      }))
    }
    
    if (uploadedFiles.length === 0) {
      toast.error("Please upload at least one image")
      return
    }
    
    setIsAnalyzing(true)
    setAnalysisProgress(0)
    
    try {
      // Prepare form data
      const formData = new FormData()
      
      // Convert age to number and ensure proper data types, auto-fill missing fields
      const processedPatientData = {
        ...patientData,
        patientInfo: {
          ...patientData.patientInfo,
          patientId: patientData.patientInfo.patientId || `PT-${Date.now()}`,
          name: patientData.patientInfo.name || 'Anonymous Patient',
          age: parseInt(patientData.patientInfo.age) || 0
        }
      }
      
      // Add sample data
      formData.append('sampleData', JSON.stringify(processedPatientData))
      
      // Add image files
      uploadedFiles.forEach(fileData => {
        formData.append('images', fileData.file)
      })
      
      // Submit with progress tracking
      // Remove authentication for testing
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 500)
      
      const response = await fetch(`${API_BASE_URL}/samples/upload-with-analysis`, {
        method: 'POST',
        body: formData
      })
      
      clearInterval(progressInterval)
      setAnalysisProgress(100)
      
      if (!response.ok) {
        let errorMessage = 'Upload failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.message || errorMessage
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      const result = await response.json()
      console.log('✅ Upload result:', result)
      
      // Store results for next components
      setMlResults(result)
      
      toast.success("Sample uploaded and analyzed successfully!")
      
      // Pass sample data to parent component - fix: use result.data.sample
      if (onSampleCreated) {
        console.log('🔍 SampleUpload: Passing sample to parent:', result.data.sample)
        onSampleCreated(result.data.sample)
      }
      
      setTimeout(() => {
        setIsAnalyzing(false)
        onNext()
      }, 1000)
      
    } catch (error) {
      console.error('Upload error:', error)
      setIsAnalyzing(false)
      setAnalysisProgress(0)
      toast.error(error.message || 'Failed to upload sample')
    }
  }

  return (
    <div className="grid gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Sample Collection & Upload</h1>
          <p className="text-muted-foreground">
            Step 1: Patient information and sample upload
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Demo Button */}
          <Dialog open={showDemoDialog} onOpenChange={(open) => { setShowDemoDialog(open); if (!open) setDemoCategory(null); }}>
            <DialogTrigger asChild>
              <Button size="sm">
                <FlaskConical className="h-4 w-4 mr-2" />
                Demo Mode
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-lg">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Microscope className="h-5 w-5" />
                  {demoCategory === null ? 'Select Analysis Type' : demoCategory === 'pneumonia' ? 'Pneumonia Detection' : 'Brain Tumor Classification'}
                </DialogTitle>
                <DialogDescription>
                  {demoCategory === null
                    ? 'Choose a detection model to demo with preloaded medical samples.'
                    : demoCategory === 'pneumonia'
                    ? 'Select a chest X-ray sample for pneumonia detection analysis.'
                    : 'Select a brain MRI sample for tumor classification analysis.'}
                </DialogDescription>
              </DialogHeader>

              {/* Step 1: Category Selection */}
              {demoCategory === null && (
                <div className="grid grid-cols-2 gap-4 py-4">
                  <Button 
                    variant="outline" 
                    className="h-36 flex-col gap-3 border-2 hover:border-blue-500 hover:bg-blue-950/30 transition-all"
                    onClick={() => setDemoCategory('pneumonia')}
                    disabled={isAnalyzing}
                  >
                    <div className="w-14 h-14 rounded-full bg-blue-500/20 flex items-center justify-center">
                      <span className="text-2xl">🫁</span>
                    </div>
                    <span className="font-semibold text-base">Pneumonia Detection</span>
                    <span className="text-xs text-muted-foreground">Chest X-Ray Analysis</span>
                  </Button>
                  <Button 
                    variant="outline" 
                    className="h-36 flex-col gap-3 border-2 hover:border-purple-500 hover:bg-purple-950/30 transition-all"
                    onClick={() => setDemoCategory('brain-tumor')}
                    disabled={isAnalyzing}
                  >
                    <div className="w-14 h-14 rounded-full bg-purple-500/20 flex items-center justify-center">
                      <span className="text-2xl">🧠</span>
                    </div>
                    <span className="font-semibold text-base">Brain Tumor Analysis</span>
                    <span className="text-xs text-muted-foreground">MRI Classification</span>
                  </Button>
                </div>
              )}

              {/* Step 2a: Pneumonia Subtypes */}
              {demoCategory === 'pneumonia' && (
                <div className="space-y-4 py-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Button 
                      variant="outline" 
                      className="h-32 flex-col gap-2 border-2 hover:border-red-500 hover:bg-red-950/30 transition-all"
                      onClick={() => handleDemoSubmit('pneumonia-positive')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                        <AlertCircle className="h-6 w-6 text-red-400" />
                      </div>
                      <span className="font-semibold">Infected</span>
                      <span className="text-xs text-muted-foreground">Pneumonia Positive</span>
                    </Button>
                    <Button 
                      variant="outline" 
                      className="h-32 flex-col gap-2 border-2 hover:border-green-500 hover:bg-green-950/30 transition-all"
                      onClick={() => handleDemoSubmit('pneumonia-negative')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                        <CheckCircle2 className="h-6 w-6 text-green-400" />
                      </div>
                      <span className="font-semibold">Normal</span>
                      <span className="text-xs text-muted-foreground">No Pneumonia</span>
                    </Button>
                  </div>
                  <Button variant="ghost" size="sm" className="w-full" onClick={() => setDemoCategory(null)}>
                    ← Back to categories
                  </Button>
                </div>
              )}

              {/* Step 2b: Brain Tumor Subtypes */}
              {demoCategory === 'brain-tumor' && (
                <div className="space-y-4 py-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Button 
                      variant="outline" 
                      className="h-28 flex-col gap-2 border-2 hover:border-red-500 hover:bg-red-950/30 transition-all"
                      onClick={() => handleDemoSubmit('glioma')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
                        <Brain className="h-5 w-5 text-red-400" />
                      </div>
                      <span className="font-semibold text-sm">Glioma</span>
                      <span className="text-xs text-muted-foreground">Malignant tumor</span>
                    </Button>
                    <Button 
                      variant="outline" 
                      className="h-28 flex-col gap-2 border-2 hover:border-orange-500 hover:bg-orange-950/30 transition-all"
                      onClick={() => handleDemoSubmit('meningioma')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-10 h-10 rounded-full bg-orange-500/20 flex items-center justify-center">
                        <Brain className="h-5 w-5 text-orange-400" />
                      </div>
                      <span className="font-semibold text-sm">Meningioma</span>
                      <span className="text-xs text-muted-foreground">Meningeal tumor</span>
                    </Button>
                    <Button 
                      variant="outline" 
                      className="h-28 flex-col gap-2 border-2 hover:border-yellow-500 hover:bg-yellow-950/30 transition-all"
                      onClick={() => handleDemoSubmit('pituitary')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-10 h-10 rounded-full bg-yellow-500/20 flex items-center justify-center">
                        <Brain className="h-5 w-5 text-yellow-400" />
                      </div>
                      <span className="font-semibold text-sm">Pituitary</span>
                      <span className="text-xs text-muted-foreground">Pituitary tumor</span>
                    </Button>
                    <Button 
                      variant="outline" 
                      className="h-28 flex-col gap-2 border-2 hover:border-green-500 hover:bg-green-950/30 transition-all"
                      onClick={() => handleDemoSubmit('no-tumor')}
                      disabled={isAnalyzing}
                    >
                      <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                        <CheckCircle2 className="h-5 w-5 text-green-400" />
                      </div>
                      <span className="font-semibold text-sm">No Tumor</span>
                      <span className="text-xs text-muted-foreground">Normal brain MRI</span>
                    </Button>
                  </div>
                  <Button variant="ghost" size="sm" className="w-full" onClick={() => setDemoCategory(null)}>
                    ← Back to categories
                  </Button>
                </div>
              )}

              <p className="text-xs text-center text-muted-foreground">
                {demoCategory === 'pneumonia' 
                  ? 'Analyzed by DenseNet121 + EfficientNet-B0 ensemble on real DICOM chest X-rays'
                  : demoCategory === 'brain-tumor'
                  ? 'Classified by EfficientNetB3 model on real brain MRI scans'
                  : 'All samples are analyzed using real ML models — no mock data'}
              </p>
            </DialogContent>
          </Dialog>
          <Badge variant="outline" className="text-sm">
            <User className="h-4 w-4 mr-2" />
            Patient Registration
          </Badge>
        </div>
      </div>

      <Tabs defaultValue="patient-info" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="patient-info">Patient Info</TabsTrigger>
          <TabsTrigger value="sample-upload">Sample Upload</TabsTrigger>
          <TabsTrigger value="review">Review & Submit</TabsTrigger>
        </TabsList>

        <TabsContent value="patient-info" className="space-y-6">
          <Card className="grid-pattern" style={{
            background: 'linear-gradient(to right, #80808033 1px, transparent 1px), linear-gradient(to bottom, #80808033 1px, transparent 1px)',
            backgroundSize: '50px 49px'
          }}>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Avatar>
                  <AvatarFallback>
                    <User className="h-4 w-4" />
                  </AvatarFallback>
                </Avatar>
                Patient Information
              </CardTitle>
              <CardDescription>
                Enter basic patient details and symptoms
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="patientId">Patient ID *</Label>
                  <Input
                    id="patientId"
                    placeholder="e.g., PT-2024-001"
                    value={patientData.patientInfo.patientId}
                    onChange={(e) => handleInputChange('patientInfo', 'patientId', e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name *</Label>
                  <Input
                    id="name"
                    placeholder="Patient full name"
                    value={patientData.patientInfo.name}
                    onChange={(e) => handleInputChange('patientInfo', 'name', e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="age">Age *</Label>
                  <Input
                    id="age"
                    type="number"
                    placeholder="25"
                    value={patientData.patientInfo.age}
                    onChange={(e) => handleInputChange('patientInfo', 'age', e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Gender *</Label>
                  <Select value={patientData.patientInfo.gender} onValueChange={(value) => handleInputChange('patientInfo', 'gender', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="contactNumber">Contact Number</Label>
                  <Input
                    id="contactNumber"
                    placeholder="+91 98765 43210"
                    value={patientData.patientInfo.contactNumber}
                    onChange={(e) => handleInputChange('patientInfo', 'contactNumber', e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="referringPhysician">Referring Physician</Label>
                  <Input
                    id="referringPhysician"
                    placeholder="Dr. Name"
                    value={patientData.patientInfo.referringPhysician}
                    onChange={(e) => handleInputChange('patientInfo', 'referringPhysician', e.target.value)}
                  />
                </div>
              </div>

              <div className="space-y-3">
                <Label className="flex items-center gap-2 text-sm font-medium">
                  Analysis Type
                  <span className="text-red-500">*</span>
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setPatientData(prev => ({ ...prev, imageType: 'tissue', clinicalInfo: { ...prev.clinicalInfo, symptoms: [] } }))}
                    className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all text-left ${
                      patientData.imageType === 'tissue'
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30'
                        : 'border-muted hover:border-blue-300 hover:bg-muted/40'
                    }`}
                  >
                    <Brain className={`h-7 w-7 ${patientData.imageType === 'tissue' ? 'text-blue-600' : 'text-muted-foreground'}`} />
                    <div>
                      <p className="text-sm font-semibold">Brain MRI</p>
                      <p className="text-xs text-muted-foreground">Tumor Detection</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">EfficientNetB3</p>
                    </div>
                    {patientData.imageType === 'tissue' && (
                      <CheckCircle2 className="h-4 w-4 text-blue-600 absolute top-2 right-2" />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => setPatientData(prev => ({ ...prev, imageType: 'pneumonia', clinicalInfo: { ...prev.clinicalInfo, symptoms: [] } }))}
                    className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all text-left ${
                      patientData.imageType === 'pneumonia'
                        ? 'border-orange-500 bg-orange-50 dark:bg-orange-950/30'
                        : 'border-muted hover:border-orange-300 hover:bg-muted/40'
                    }`}
                  >
                    <Stethoscope className={`h-7 w-7 ${patientData.imageType === 'pneumonia' ? 'text-orange-600' : 'text-muted-foreground'}`} />
                    <div>
                      <p className="text-sm font-semibold">Chest X-ray</p>
                      <p className="text-xs text-muted-foreground">Pneumonia Detection</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">DenseNet121 + EffNet-B0</p>
                    </div>
                    {patientData.imageType === 'pneumonia' && (
                      <CheckCircle2 className="h-4 w-4 text-orange-600 absolute top-2 right-2" />
                    )}
                  </button>
                </div>
                {!patientData.imageType && (
                  <p className="text-xs text-red-500">Please select an analysis type before uploading.</p>
                )}
              </div>

              <div className="space-y-3">
                <Label>Clinical Diagnosis / Reason for Scan</Label>
                <Input
                  placeholder={patientData.imageType === 'pneumonia' ? 'e.g., Suspected community-acquired pneumonia' : 'e.g., Suspected intracranial lesion'}
                  value={patientData.clinicalInfo.clinicalDiagnosis}
                  onChange={(e) => handleInputChange('clinicalInfo', 'clinicalDiagnosis', e.target.value)}
                />
              </div>

              <div className="space-y-3">
                <Label>Medical History (optional)</Label>
                <Input
                  placeholder={patientData.imageType === 'pneumonia' ? 'e.g., Asthma, COPD, recent surgery' : 'e.g., Previous stroke, family history of tumors'}
                  value={patientData.clinicalInfo.medicalHistory}
                  onChange={(e) => handleInputChange('clinicalInfo', 'medicalHistory', e.target.value)}
                />
              </div>

              <div className="space-y-3">
                <Label>
                  Symptoms (Select all that apply)
                  {patientData.imageType && (
                    <span className="ml-2 text-xs text-muted-foreground font-normal">
                      — {patientData.imageType === 'pneumonia' ? 'Respiratory symptoms' : 'Neurological symptoms'}
                    </span>
                  )}
                </Label>
                {!patientData.imageType ? (
                  <p className="text-xs text-muted-foreground">Please select an analysis type above to see relevant symptoms.</p>
                ) : (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {symptoms.map((symptom) => (
                      <div key={symptom} className="flex items-center space-x-2">
                        <Checkbox
                          id={symptom}
                          checked={patientData.clinicalInfo.symptoms.includes(symptom)}
                          onCheckedChange={() => handleSymptomChange(symptom)}
                        />
                        <Label htmlFor={symptom} className="text-sm">
                          {symptom}
                        </Label>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sample-upload" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5 text-blue-500" />
                Medical Image Upload
                {isAnalyzing && (
                  <Badge variant="secondary" className="ml-2">
                    <Brain className="h-3 w-3 mr-1" />
                    AI Analyzing...
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Upload medical images for AI-powered analysis and radiologist review
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                <Upload className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
                <Label htmlFor="image-upload" className="cursor-pointer">
                  <span className="text-sm font-medium">Click to upload medical images</span>
                  <br />
                  <span className="text-xs text-muted-foreground">
                    JPEG, PNG, BMP up to 50MB each (max 10 files)
                  </span>
                  <br />
                  <span className="text-xs font-medium text-blue-600">
                    🫁 DICOM (.dcm) also supported for Pneumonia Detection
                  </span>
                </Label>
                <Input
                  id="image-upload"
                  type="file"
                  multiple
                  accept="image/jpeg,image/png,image/bmp,.jpg,.jpeg,.png,.bmp,.dcm,application/dicom"
                  className="hidden"
                  onChange={handleFileUpload}
                />
              </div>

              {/* Analysis Progress */}
              {isAnalyzing && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">AI Analysis in Progress...</span>
                  </div>
                  <Progress value={analysisProgress} className="w-full" />
                  <p className="text-xs text-muted-foreground">
                    Processing images with {patientData.imageType === 'pneumonia' ? 'DenseNet121 + EfficientNet-B0 Ensemble' : 'EfficientNetB3 brain tumor detection'} model
                  </p>
                </div>
              )}

              {/* Uploaded Files Display */}
              {uploadedFiles.length > 0 && (
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Selected Files ({uploadedFiles.length})</Label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-60 overflow-y-auto">
                    {uploadedFiles.map((file) => (
                      <div key={file.id} className={`flex items-center justify-between p-3 border rounded-lg ${file.isDicom ? 'border-blue-200 bg-blue-50/50' : ''}`}>
                        <div className="flex items-center gap-3">
                          {file.isDicom ? (
                            // DICOM files can't be previewed in browser
                            <div className="w-10 h-10 rounded bg-blue-100 border border-blue-300 flex flex-col items-center justify-center">
                              <span className="text-[9px] font-bold text-blue-700 leading-tight">DICOM</span>
                              <span className="text-[8px] text-blue-500">.dcm</span>
                            </div>
                          ) : file.preview ? (
                            <img 
                              src={file.preview} 
                              alt={file.name}
                              className="w-10 h-10 object-cover rounded"
                            />
                          ) : null}
                          <div>
                            <p className="text-sm font-medium truncate max-w-32">{file.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                              {file.isDicom && <span className="ml-1 text-blue-600 font-medium">• Chest X-ray DICOM</span>}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(file.id)}
                          className="text-red-500 hover:text-red-700"
                        >
                          ×
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>



        <TabsContent value="review" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Review & Submit</CardTitle>
              <CardDescription>
                Please review all information before submitting
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm font-medium">Patient ID</Label>
                  <p className="text-sm text-muted-foreground">{patientData.patientInfo.patientId || 'Not provided'}</p>
                </div>
                <div>
                  <Label className="text-sm font-medium">Full Name</Label>
                  <p className="text-sm text-muted-foreground">{patientData.patientInfo.name || 'Not provided'}</p>
                </div>
                <div>

                  <Label className="text-sm font-medium">Analysis Type</Label>
                  <p className="text-sm text-muted-foreground">
                    {patientData.imageType === 'tissue' ? (
                      <span className="flex items-center gap-1">
                        <Brain className="h-3 w-3" />
                        Brain MRI (Tumor Detection) — EfficientNetB3
                      </span>
                    ) : patientData.imageType === 'pneumonia' ? (
                      <span className="flex items-center gap-1">
                        <Stethoscope className="h-3 w-3" />
                        Chest X-ray (Pneumonia Detection) — DenseNet121 + EfficientNet-B0
                      </span>
                    ) : 'Not selected'}
                  </p>
                </div>
                <div>
                  <Label className="text-sm font-medium">Files Uploaded</Label>
                  <p className="text-sm text-muted-foreground">{uploadedFiles.length} files</p>
                </div>
              </div>

              {patientData.clinicalInfo.symptoms.length > 0 && (
                <div>
                  <Label className="text-sm font-medium">Symptoms</Label>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {patientData.clinicalInfo.symptoms.map(symptom => (
                      <Badge key={symptom} variant="outline" className="text-xs">
                        {symptom}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {mlResults && (
                <Alert>
                  <Brain className="h-4 w-4" />
                  <AlertTitle>AI Analysis Complete</AlertTitle>
                  <AlertDescription>
                    {mlResults.mlAnalysis ? 
                      `${mlResults.mlAnalysis.imagesAnalyzed} images analyzed. ${mlResults.mlAnalysis.aiInsights?.highRiskImages || 0} high-risk findings detected.` :
                      'Sample processing completed successfully.'
                    }
                  </AlertDescription>
                </Alert>
              )}

              {!mlResults && (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Important</AlertTitle>
                  <AlertDescription>
                    Please ensure all patient information is accurate and all required samples are uploaded before proceeding.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
            <CardFooter>
              {uploadedFiles.length > 0 && !patientData.imageType && (
                <p className="text-xs text-amber-600 font-medium pb-2 text-center">
                  ⚠️ Please select an analysis type (Brain MRI or Chest X-ray) in the Patient Info tab before submitting.
                </p>
              )}
              <Button 
                onClick={handleSubmit} 
                className="w-full"
                disabled={isAnalyzing || uploadedFiles.length === 0 || !patientData.imageType}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing & Analyzing...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    Submit Sample for Analysis
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}