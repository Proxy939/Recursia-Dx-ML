import OpenAI from 'openai';

// Lazy-initialize OpenAI client
let client = null;

function getClient() {
    if (!client && process.env.OPENAI_API_KEY) {
        client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    }
    return client;
}

/**
 * Core helper — call GPT-4o-mini with a system + user prompt, return text.
 */
async function callGPT(systemPrompt, userPrompt) {
    const ai = getClient();
    if (!ai) throw new Error('OPENAI_API_KEY not configured');

    const completion = await ai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user',   content: userPrompt }
        ],
        temperature: 0.3,
        max_tokens: 1024
    });

    return completion.choices[0].message.content.trim();
}

// ── Modality helpers ───────────────────────────────────────────────────────────

function resolveModality(sampleType) {
    const t = (sampleType || '').toLowerCase();
    if (t.includes('pneumonia') || t.includes('chest') || t.includes('xray') || t.includes('x-ray')) {
        return 'pneumonia';
    }
    return 'brain_tumor';
}

function modalityLabel(modality) {
    return modality === 'pneumonia'
        ? 'Chest X-Ray (Pneumonia Detection)'
        : 'Brain MRI (Tumor Detection)';
}

// ── Full Report ───────────────────────────────────────────────────────────────

/**
 * Generate a comprehensive radiology report using OpenAI GPT-4o-mini.
 */
export async function generateFullReport(sample, mlResults) {
    if (!getClient()) {
        console.warn('⚠️ OPENAI_API_KEY not set - using fallback');
        return { success: false, error: 'OpenAI API key not configured' };
    }

    const modality = resolveModality(sample.sampleType);
    const isPneumonia = modality === 'pneumonia';

    try {
        console.log('🤖 Calling OpenAI GPT-4o-mini for full radiology report...');

        const system = `You are an expert radiologist AI assistant generating professional clinical imaging reports. 
Write in a formal, precise medical style. Always state that AI analysis requires verification by a qualified radiologist.`;

        // Build heatmap context string if available
        const heatmapCount = mlResults.heatmaps?.length || 0;
        const affectedArea = mlResults.affectedAreaPct ? `${mlResults.affectedAreaPct.toFixed(1)}%` : null;
        const heatmapContext = heatmapCount > 0
          ? `\nGRAD-CAM HEATMAP DATA:\n- Heatmap images available: ${heatmapCount}\n${affectedArea ? `- Affected area (from heatmap): ${affectedArea}\n` : ''}- Heatmap type: Gradient-weighted Class Activation Map (highlights model attention regions)`
          : '';

        const user = `Generate a radiology report for the following imaging study.

STUDY INFORMATION:
- Sample ID: ${sample.sampleId || sample._id}
- Imaging Modality: ${modalityLabel(modality)}
- Collection Date: ${sample.sampleInfo?.collectionDate || 'Not specified'}

PATIENT:
- Name: ${sample.patientInfo?.name || 'Anonymous'}
- Age: ${sample.patientInfo?.age || 'Unknown'}
- Gender: ${sample.patientInfo?.gender || 'Unknown'}

AI ANALYSIS RESULTS:
- Total Images Analyzed: ${mlResults.totalImages || 0}
- Finding: ${mlResults.isPositive ? (isPneumonia ? 'PNEUMONIA DETECTED' : 'TUMOR DETECTED') : 'NORMAL — No Pathology Detected'}
- ${isPneumonia ? 'Pneumonia' : 'Tumor'} Probability: ${(mlResults.tumorProbability || 0).toFixed(1)}%
- Model Confidence: ${(mlResults.confidence || 0).toFixed(1)}%
- Risk Level: ${mlResults.riskLevel || 'Unknown'}
${isPneumonia ? `- Severity: ${mlResults.severity || 'Unknown'}` : `- Tumor Class: ${mlResults.tumorClass || 'Unknown'}`}
${heatmapContext}

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "clinicalSummary": "2-3 sentence professional summary of imaging findings",
  "interpretation": "Detailed radiological interpretation of AI results (3-4 sentences, use imaging terminology)",
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "imagingFindings": "Description of key imaging characteristics observed",
  "heatmapInterpretation": ${heatmapCount > 0 ? `"1-2 sentence interpretation of what the Grad-CAM attention map indicates about model focus regions"` : `"No heatmap data available"`},
  "conclusion": "Final radiological impression (1-2 sentences)"
}`;

        const text = await callGPT(system, user);

        let reportContent;
        try {
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            reportContent = JSON.parse(jsonMatch ? jsonMatch[0] : text);
        } catch {
            reportContent = {
                clinicalSummary: text.substring(0, 300),
                interpretation: 'AI-generated interpretation available.',
                recommendations: ['Consult with a radiologist for confirmation'],
                imagingFindings: 'See detailed analysis above.',
                conclusion: 'AI analysis complete. Radiologist review recommended.'
            };
        }

        return {
            success: true,
            generatedBy: 'OpenAI GPT-4o-mini',
            timestamp: new Date().toISOString(),
            content: reportContent
        };

    } catch (error) {
        console.error('OpenAI full report error:', error.message);
        return { success: false, error: error.message, generatedBy: 'Error' };
    }
}

// ── Clinical Summary ──────────────────────────────────────────────────────────

/**
 * Generate a short clinical summary from ML analysis results.
 */
export async function generateClinicalSummary(mlResults, patientInfo = {}) {
    if (!getClient()) {
        return generateFallbackSummary(mlResults);
    }

    const modality = resolveModality(mlResults.sampleType);
    const isPneumonia = modality === 'pneumonia';

    try {
        console.log('🤖 Calling OpenAI GPT-4o-mini for clinical summary...');

        const system = `You are a medical AI assistant writing concise clinical summaries for radiology reports. 
Use professional medical language appropriate for physicians.`;

        const user = `Write a 2-3 sentence clinical summary for this imaging study.

Imaging Type: ${modalityLabel(modality)}
Images Analyzed: ${mlResults.totalImages || 0}
Finding: ${mlResults.isPositive ? (isPneumonia ? 'Pneumonia Detected' : 'Tumor Detected') : 'Normal'}
${isPneumonia ? 'Pneumonia' : 'Tumor'} Probability: ${(mlResults.tumorProbability || 0).toFixed(1)}%
AI Confidence: ${((mlResults.averageConfidence || mlResults.confidence || 0) * (mlResults.averageConfidence > 1 ? 1 : 100)).toFixed(1)}%
Risk Level: ${mlResults.overallRisk || mlResults.riskLevel || 'Unknown'}
Patient Age: ${patientInfo.age || 'Unknown'}, Gender: ${patientInfo.gender || 'Unknown'}

Use phrases like "findings are consistent with", "AI-assisted analysis suggests". 
Include a recommendation if abnormal. Output plain text only (no JSON, no markdown).`;

        const summary = await callGPT(system, user);

        return {
            success: true,
            summary,
            generatedBy: 'OpenAI GPT-4o-mini',
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        console.error('OpenAI clinical summary error:', error.message);
        return generateFallbackSummary(mlResults);
    }
}

// ── Recommendations ───────────────────────────────────────────────────────────

/**
 * Generate 3 clinical follow-up recommendations.
 */
export async function generateRecommendations(mlResults) {
    if (!getClient()) {
        return generateFallbackRecommendations(mlResults);
    }

    const modality = resolveModality(mlResults.sampleType);
    const isPneumonia = modality === 'pneumonia';

    try {
        console.log('🤖 Calling OpenAI GPT-4o-mini for recommendations...');

        const system = `You are a clinical AI assistant providing evidence-based radiology follow-up recommendations.`;

        const user = `Based on these radiology AI results, provide exactly 3 concise clinical recommendations.

Imaging: ${modalityLabel(modality)}
Finding: ${mlResults.isPositive ? (isPneumonia ? 'Pneumonia Detected' : 'Tumor Detected') : 'Normal'}
Confidence: ${(mlResults.confidence || 0).toFixed(1)}%
Risk: ${mlResults.riskLevel || 'Unknown'}

Return ONLY a valid JSON array of 3 strings (no markdown, no explanation):
["Recommendation 1", "Recommendation 2", "Recommendation 3"]`;

        const text = await callGPT(system, user);

        let recommendations;
        try {
            const arrMatch = text.match(/\[[\s\S]*\]/);
            recommendations = JSON.parse(arrMatch ? arrMatch[0] : text);
        } catch {
            recommendations = generateFallbackRecommendations(mlResults).recommendations;
        }

        return { success: true, recommendations, generatedBy: 'OpenAI GPT-4o-mini' };
    } catch (error) {
        console.error('OpenAI recommendations error:', error.message);
        return generateFallbackRecommendations(mlResults);
    }
}

// ── Fallbacks ─────────────────────────────────────────────────────────────────

function generateFallbackSummary(mlResults) {
    const isPositive = mlResults.isPositive || (mlResults.malignantDetections || 0) > 0;
    const rawConf = mlResults.averageConfidence || mlResults.confidence || 0;
    const confidence = (rawConf > 1 ? rawConf : rawConf * 100).toFixed(1);
    const risk = mlResults.overallRisk || mlResults.riskLevel || 'Unknown';
    const modality = resolveModality(mlResults.sampleType);
    const finding = modality === 'pneumonia'
        ? (isPositive ? 'consolidative opacities consistent with pneumonia' : 'clear lung fields with no evidence of pneumonia')
        : (isPositive ? 'intracranial mass lesion requiring further evaluation' : 'no evidence of intracranial neoplasm');

    const summary = isPositive
        ? `AI-assisted ${modalityLabel(modality)} analysis identifies ${finding} with ${confidence}% model confidence. Risk assessment: ${risk}. Clinical correlation and radiologist review are strongly recommended.`
        : `AI-assisted ${modalityLabel(modality)} analysis demonstrates ${finding}. Model confidence: ${confidence}%. Risk level: ${risk}. Routine follow-up per clinical guidelines is advised.`;

    return { success: true, summary, generatedBy: 'Fallback', timestamp: new Date().toISOString() };
}

function generateFallbackRecommendations(mlResults) {
    const isPositive = mlResults.isPositive || (mlResults.malignantDetections || 0) > 0;
    const modality = resolveModality(mlResults.sampleType);
    const isPneumonia = modality === 'pneumonia';

    const recommendations = isPositive
        ? isPneumonia
            ? [
                'Initiate appropriate antibiotic therapy per local guidelines and clinical severity',
                'Consider repeat chest X-ray in 4-6 weeks to confirm radiological clearance',
                'Escalate to CT chest if clinical deterioration or atypical features are present'
              ]
            : [
                'Urgent MRI brain with contrast for detailed lesion characterisation',
                'Neurosurgery and neuro-oncology multidisciplinary team referral',
                'Assess for raised intracranial pressure and initiate appropriate management'
              ]
        : isPneumonia
            ? [
                'No immediate antibiotic treatment required based on current imaging',
                'Continue routine clinical assessment and symptom monitoring',
                'Repeat imaging only if clinical symptoms persist or worsen'
              ]
            : [
                'No immediate neurosurgical intervention required based on current imaging',
                'Routine neurological follow-up as per clinical indication',
                'Repeat MRI in 6-12 months if clinically indicated'
              ];

    return { success: true, recommendations, generatedBy: 'Fallback' };
}
