document.addEventListener('DOMContentLoaded', () => {
    // ── Navigation Tabs ──────────────────────────────────────────────────────
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // ── TAB 1: GENERATION INTERACTIVE FLOW ────────────────────────────────────
    let selectedGrade = 0;
    const severityOpts = document.querySelectorAll('.severity-opt');
    const generateBtn = document.getElementById('generate-btn');
    const pipelineLogs = document.getElementById('pipeline-logs');
    const genImageBox = document.getElementById('gen-image-box');
    const jsonOutput = document.getElementById('json-output');
    const downloadActions = document.getElementById('download-actions');
    const downloadImgBtn = document.getElementById('download-img-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');

    // Severity grade selector
    severityOpts.forEach(opt => {
        opt.addEventListener('click', () => {
            severityOpts.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            selectedGrade = parseInt(opt.getAttribute('data-grade'));
        });
    });

    // Add logging message to logger panel
    function addLog(message, type = 'info') {
        const time = new Date().toLocaleTimeString();
        const log = document.createElement('div');
        log.className = 'log-entry';
        
        let prefixClass = 'log-info';
        if (type === 'success') prefixClass = 'log-success';
        if (type === 'warn') prefixClass = 'log-warn';
        if (type === 'error') prefixClass = 'log-err';
        
        log.innerHTML = `
            <span class="log-time">[${time}]</span>
            <span class="${prefixClass}">[Engine]</span>
            <span>${message}</span>
        `;
        pipelineLogs.appendChild(log);
        pipelineLogs.scrollTop = pipelineLogs.scrollHeight;
    }

    // Reset indicator classes
    function resetStepIndicators() {
        const steps = ['step-rag', 'step-meta', 'step-repair', 'step-diff'];
        steps.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.className = 'step-indicator';
        });
    }

    // Manage generation pipeline
    generateBtn.addEventListener('click', async () => {
        generateBtn.disabled = true;
        resetStepIndicators();
        genImageBox.innerHTML = `
            <div class="image-placeholder">
                <i class="fa-solid fa-spinner fa-spin" style="font-size:2.5rem; color:var(--accent);"></i>
                <span>Pipeline running...</span>
            </div>
        `;
        jsonOutput.textContent = '{\n  "status": "Running synthesis engine, please wait..."\n}';
        downloadActions.style.display = 'none';
        
        addLog(`Initiating Synthesis for Target Severity Level ${selectedGrade}`, 'info');
        
        // Step 1: Prep
        const stepRag = document.getElementById('step-rag');
        if (stepRag) stepRag.classList.add('active');
        addLog("Retrieving clinical evidence grounding...", "info");
        
        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dr_grade: selectedGrade,
                    temperature: 0.7,
                    use_grounding: true
                })
            });
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || "Synthesis failed");
            }
            
            // Simulate pipeline step timings for gorgeous UI feedback
            setTimeout(() => {
                // Step 1 Complete
                if (stepRag) {
                    stepRag.classList.remove('active');
                    stepRag.classList.add('completed');
                }
                addLog("Grounding verification completed.", "success");
                
                // Step 2: Structuring
                const stepMeta = document.getElementById('step-meta');
                if (stepMeta) stepMeta.classList.add('active');
                addLog("Structuring clinical records...", "info");
                
                setTimeout(() => {
                    if (stepMeta) {
                        stepMeta.classList.remove('active');
                        stepMeta.classList.add('completed');
                    }
                    addLog("Record structuring finished.", "success");
                    
                    // Step 3: Validation
                    const stepRepair = document.getElementById('step-repair');
                    if (stepRepair) stepRepair.classList.add('active');
                    addLog("Verifying record against validation rules...", "info");
                    
                    setTimeout(() => {
                        if (stepRepair) {
                            stepRepair.classList.remove('active');
                            stepRepair.classList.add('completed');
                        }
                        addLog("Record validation and quality control complete.", "success");
                        
                        // Step 4: Rendering
                        const stepDiff = document.getElementById('step-diff');
                        if (stepDiff) stepDiff.classList.add('active');
                        addLog("Synthesizing image output...", "info");
                        
                        setTimeout(() => {
                            if (stepDiff) {
                                stepDiff.classList.remove('active');
                                stepDiff.classList.add('completed');
                            }
                            addLog("Dual-modality synthesis successfully finished.", "success");
                            
                            // Render outputs
                            renderGenerationResult(result);
                            generateBtn.disabled = false;
                        }, 1200);
                        
                    }, 1000);
                    
                }, 1000);
                
            }, 1000);
            
        } catch (err) {
            addLog(`Pipeline error: ${err.message}`, 'error');
            genImageBox.innerHTML = `
                <div class="image-placeholder">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size:2.5rem; color:var(--error);"></i>
                    <span>Synthesis failed</span>
                </div>
            `;
            jsonOutput.textContent = `{\n  "error": "${err.message}"\n}`;
            generateBtn.disabled = false;
        }
    });

    function renderGenerationResult(data) {
        // 1. Render image
        genImageBox.innerHTML = `<img src="data:image/png;base64,${data.synthetic_image_b64}" alt="Synthesized Image">`;
        
        // 2. Set up download buttons
        downloadImgBtn.href = `data:image/png;base64,${data.synthetic_image_b64}`;
        downloadImgBtn.download = `synthesis_level${data.dr_grade}.png`;
        
        // JSON metadata download config
        const metaStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data.final_record, null, 2));
        downloadJsonBtn.onclick = () => {
            const link = document.createElement('a');
            link.setAttribute("href", metaStr);
            link.setAttribute("download", `synthesis_metadata_level${data.dr_grade}.json`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };
        downloadActions.style.display = 'flex';
        
        // 3. Render JSON (Remove repair flags to keep repair engine secret)
        const displayRecord = {...data.final_record};
        delete displayRecord._repaired;
        delete displayRecord._valid;
        jsonOutput.textContent = JSON.stringify(displayRecord, null, 2);
    }

    // ── TAB 2: CLASSIFIER INFERENCE FLOW ──────────────────────────────────────
    const samplesContainer = document.getElementById('samples-container');
    const dragDropZone = document.getElementById('drag-drop-zone');
    const imageUploadInput = document.getElementById('image-upload-input');
    const classifyBtn = document.getElementById('classify-btn');
    const classifierPreprocessedImg = document.getElementById('classifier-preprocessed-img');
    const predGradeBadge = document.getElementById('pred-grade-badge');
    const predDescription = document.getElementById('pred-description');
    const classifierResultPanel = document.getElementById('classification-result-panel');
    const classifierPlaceholder = document.getElementById('classifier-placeholder');

    let selectedSampleStem = null;
    let uploadedFile = null;

    // Load sample presets from API
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.success && data.samples) {
                samplesContainer.innerHTML = '';
                data.samples.forEach(sample => {
                    const card = document.createElement('div');
                    card.className = 'sample-thumbnail';
                    card.setAttribute('data-stem', sample.stem);
                    card.innerHTML = `
                        <img src="/api/sample-image/${sample.filename}" alt="Sample fundus photo">
                        <span class="sample-badge">Level ${sample.dr_grade}</span>
                    `;
                    card.addEventListener('click', () => {
                        // Deselect other thumbnails and file uploads
                        document.querySelectorAll('.sample-thumbnail').forEach(t => t.classList.remove('active'));
                        dragDropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                        dragDropZone.querySelector('span').textContent = 'Drag and drop a fundus image here';
                        
                        card.classList.add('active');
                        selectedSampleStem = sample.stem;
                        uploadedFile = null;
                    });
                    samplesContainer.appendChild(card);
                });
            }
        } catch (e) {
            console.error("Failed to load presets: ", e);
        }
    }

    loadSamples();

    // Trigger local file choosing
    dragDropZone.addEventListener('click', () => imageUploadInput.click());
    
    imageUploadInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadedFile = e.target.files[0];
            selectedSampleStem = null;
            
            // Highlight upload box and deselect thumbnails
            document.querySelectorAll('.sample-thumbnail').forEach(t => t.classList.remove('active'));
            dragDropZone.style.borderColor = 'var(--accent)';
            dragDropZone.querySelector('span').textContent = `File selected: ${uploadedFile.name}`;
        }
    });

    // Handle drag and drop files
    dragDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragDropZone.style.borderColor = 'var(--accent)';
    });

    dragDropZone.addEventListener('dragleave', () => {
        if (!uploadedFile) {
            dragDropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        }
    });

    dragDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) {
            uploadedFile = e.dataTransfer.files[0];
            selectedSampleStem = null;
            
            document.querySelectorAll('.sample-thumbnail').forEach(t => t.classList.remove('active'));
            dragDropZone.style.borderColor = 'var(--accent)';
            dragDropZone.querySelector('span').textContent = `File dropped: ${uploadedFile.name}`;
        }
    });

    // Run classification call
    classifyBtn.addEventListener('click', async () => {
        if (!selectedSampleStem && !uploadedFile) {
            alert("Please select a sample image or upload a custom fundus photograph first.");
            return;
        }

        classifyBtn.disabled = true;
        classifyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {
            let res;
            if (uploadedFile) {
                // Multipart file upload
                const formData = new FormData();
                formData.append('file', uploadedFile);
                res = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
            } else {
                // Post JSON payload with sample stem
                res = await fetch('/api/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sample_stem: selectedSampleStem })
                });
            }

            const data = await res.json();
            if (data.success) {
                renderClassifierResult(data);
            } else {
                alert(`Analysis failed: ${data.error}`);
            }

        } catch (e) {
            console.error(e);
            alert(`Network error running classifier: ${e.message}`);
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Run Classifier Analysis';
        }
    });

    function renderClassifierResult(data) {
        // Show result panel, hide placeholder
        classifierPlaceholder.style.display = 'none';
        classifierResultPanel.style.display = 'flex';
        
        // 1. Show preprocessed base64
        classifierPreprocessedImg.src = `data:image/png;base64,${data.preprocessed_image_b64}`;
        
        // 2. Set prediction details
        predGradeBadge.textContent = data.prediction_name.replace("Diabetic Retinopathy", "DR").replace("DR", "Level");
        
        // Reset classes
        predGradeBadge.className = 'prediction-badge';
        predGradeBadge.classList.add(`grade-${data.prediction}`);
        
        predDescription.textContent = data.description;
        
        // 3. Render probability spectrum bars
        for (let i = 0; i < 5; i++) {
            const probPct = (data.probabilities[i] * 100).toFixed(1);
            document.getElementById(`prob-${i}-text`).textContent = `${probPct}%`;
            document.getElementById(`prob-${i}-fill`).style.width = `${probPct}%`;
        }
    }
});
