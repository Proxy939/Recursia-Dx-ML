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

export function SampleUpload({ onNext, onSampleCreated }) {
  const [patientData, setPatientData] = useState({
    patientInfo: {
      patientId: '',
      name: '',
      age: '',
      gender: '',
      dateOfBirth: '',
      contactNumber: '',
      address: ''
    },
    imageType: 'tissue', // auto-detected: tissue (brain MRI) or pneumonia (chest X-ray)
    specimenDetails: {
      organ: '',
      site: '',
      size: '',
      color: '',
      consistency: ''
    },
    clinicalInfo: {
      clinicalDiagnosis: '',
      symptoms: [],
      duration: '',
      urgency: 'Routine'
    },
    collectionInfo: {
      collectionDate: new Date().toISOString().split('T')[0],
      collectionTime: new Date().toTimeString().split(' ')[0].substring(0, 5),
      collectionMethod: '',
      transportConditions: ''
    }
  })
  
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [mlResults, setMlResults] = useState(null)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [showDemoDialog, setShowDemoDialog] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(false)

  const symptoms = [
    'Fatigue', 'Fever', 'Weight Loss', 'Night Sweats', 
    'Unusual Bleeding', 'Persistent Cough', 'Difficulty Swallowing',
    'Enlarged Lymph Nodes', 'Bone Pain', 'Easy Bruising'
  ]


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
    
    // Validate file types (no TIFF, no SVS/NDPI)
    const validFiles = files.filter(file => {
      const ext = file.name.toLowerCase().split('.').pop()
      if (['tiff', 'tif', 'svs', 'ndpi'].includes(ext)) {
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
    
    // Auto-detect image type from filename
    const allFiles = [...uploadedFiles, ...newFiles]
    const fileNames = allFiles.map(f => f.name.toLowerCase()).join(' ')
    // DICOM files from chest X-rays → always route to pneumonia detection
    const hasDicom = allFiles.some(f => f.isDicom)
    const isPneumonia = hasDicom ||
                        fileNames.includes('xray') || fileNames.includes('x-ray') || 
                        fileNames.includes('chest') || fileNames.includes('pneumonia') || 
                        fileNames.includes('lung') || fileNames.includes('cxr')
    const detectedType = isPneumonia ? 'pneumonia' : 'tissue'
    
    setPatientData(prev => ({ ...prev, imageType: detectedType }))
    
    const typeLabel = isPneumonia ? 'Chest X-ray / DICOM (Pneumonia Detection)' : 'Brain MRI (Tumor Detection)'
    toast.success(`${validFiles.length} file(s) selected — auto-detected as ${typeLabel}`)
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

  // Handle Demo Mode - Process preloaded WSI samples
  const handleDemoSubmit = async (demoType) => {
    setShowDemoDialog(false)
    setIsDemoMode(true)
    setIsAnalyzing(true)
    setAnalysisProgress(0)
    
    // Auto-fill demo patient data
    setPatientData(prev => ({
      ...prev,
      patientInfo: {
        patientId: `DEMO-${Date.now()}`,
        name: demoType === 'tumor' ? 'Demo Patient (Tumor Sample)' : 'Demo Patient (Normal Sample)',
        age: '45',
        gender: 'Female'
      },
      imageType: 'tissue'
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
      const response = await fetch(`http://localhost:5000/api/samples/demo-analysis?type=${demoType}`, {
        method: 'POST'
      })
      
      clearInterval(progressInterval)
      setAnalysisProgress(100)
      
      if (!response.ok) {
        throw new Error('Demo analysis failed')
      }
      
      const result = await response.json()
      console.log('✅ Demo result:', result)
      
      setMlResults(result)
      toast.success(`Demo ${demoType} sample analyzed successfully!`)
      
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
      
      const response = await fetch('http://localhost:5000/api/samples/upload-with-analysis', {
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
          <Dialog open={showDemoDialog} onOpenChange={setShowDemoDialog}>
            <DialogTrigger asChild>
              <Button size="sm">
                <FlaskConical className="h-4 w-4 mr-2" />
                Demo Mode
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Microscope className="h-5 w-5" />
                  Select Demo Sample
                </DialogTitle>
                <DialogDescription>
                  Choose a preloaded sample to test the Brain Tumor detection model.
                </DialogDescription>
              </DialogHeader>
              <div className="grid grid-cols-2 gap-4 py-4">
                <Button 
                  variant="outline" 
                  className="h-32 flex-col gap-2 border-2 hover:border-red-500 hover:bg-red-50"
                  onClick={() => handleDemoSubmit('tumor')}
                  disabled={isAnalyzing}
                >
                  <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                    <AlertCircle className="h-6 w-6 text-red-600" />
                  </div>
                  <span className="font-semibold">Tumor Sample</span>
                  <span className="text-xs text-muted-foreground">Malignant tissue</span>
                </Button>
                <Button 
                  variant="outline" 
                  className="h-32 flex-col gap-2 border-2 hover:border-green-500 hover:bg-green-50"
                  onClick={() => handleDemoSubmit('non-tumor')}
                  disabled={isAnalyzing}
                >
                  <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
                    <CheckCircle2 className="h-6 w-6 text-green-600" />
                  </div>
                  <span className="font-semibold">Normal Sample</span>
                  <span className="text-xs text-muted-foreground">Healthy tissue</span>
                </Button>
              </div>
              <p className="text-xs text-center text-muted-foreground">
                Demo samples are processed using the EfficientNetB3 brain tumor model
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
                  <Label>Blood Group</Label>
                  <Select onValueChange={(value) => setPatientData(prev => ({...prev, bloodGroup: value}))}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select blood group" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="A+">A+</SelectItem>
                      <SelectItem value="A-">A-</SelectItem>
                      <SelectItem value="B+">B+</SelectItem>
                      <SelectItem value="B-">B-</SelectItem>
                      <SelectItem value="AB+">AB+</SelectItem>
                      <SelectItem value="AB-">AB-</SelectItem>
                      <SelectItem value="O+">O+</SelectItem>
                      <SelectItem value="O-">O-</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-3">
                <Label className="flex items-center gap-2">
                  Image Type
                  <Badge variant="outline" className="text-xs">
                    Auto-Detected
                  </Badge>
                </Label>
                <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/30">
                  {patientData.imageType === 'pneumonia' ? (
                    <>
                      <Stethoscope className="h-5 w-5 text-orange-500" />
                      <div>
                        <p className="text-sm font-medium">Chest X-ray (Pneumonia Detection)</p>
                        <p className="text-xs text-muted-foreground">DenseNet121 + EfficientNet-B0 Ensemble</p>
                      </div>
                    </>
                  ) : (
                    <>
                      <Brain className="h-5 w-5 text-blue-500" />
                      <div>
                        <p className="text-sm font-medium">Brain MRI (Tumor Detection)</p>
                        <p className="text-xs text-muted-foreground">EfficientNetB3 model</p>
                      </div>
                    </>
                  )}
                  <Badge variant="secondary" className="ml-auto text-xs">Auto</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Image type is auto-detected from filenames. Include keywords like "xray", "chest", or "lung" for pneumonia detection.
                  You can also override it below.
                </p>
                <Select value={patientData.imageType} onValueChange={(value) => setPatientData(prev => ({...prev, imageType: value}))}>
                  <SelectTrigger className="text-xs h-8">
                    <SelectValue placeholder="Override detection" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tissue">Brain MRI / Tissue (Tumor Detection)</SelectItem>
                    <SelectItem value="pneumonia">Chest X-ray (Pneumonia Detection)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label>Clinical Diagnosis</Label>
                <Input
                  placeholder="Enter preliminary diagnosis"
                  value={patientData.clinicalInfo.clinicalDiagnosis}
                  onChange={(e) => handleInputChange('clinicalInfo', 'clinicalDiagnosis', e.target.value)}
                />
              </div>

              <div className="space-y-3">
                <Label>Symptoms (Select all that apply)</Label>
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
                Upload medical images for AI-powered analysis and pathologist review
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

                  <Label className="text-sm font-medium">Image Type (Auto-Detected)</Label>
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
              <Button 
                onClick={handleSubmit} 
                className="w-full"
                disabled={isAnalyzing || uploadedFiles.length === 0}
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