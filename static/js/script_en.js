// Global variables
let probabilityChart = null;
let waveformChart = null;
let lossChart = null;
let accuracyChart = null;
let splitVisualizationChart = null;

// QTBFS visualization globals
let qtbfsVisualData = null;
let qtbfsChart = null;

// Training data
let trainingData = {
    epochs: [],
    losses: [],
    accuracies: []
};

// File references
let selectedDatasetFile = null;
let selectedModelFile = null;
let selectedDataFile = null;
let selectedVisFile = null;
let selectedSplitFile = null;
let state0Files = [];
let currentFiles = [];

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = 'notification ' + type;
    notification.classList.add('show');

    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Switch tab
function switchTab(tabName) {
    document.querySelectorAll('.side-nav li').forEach(tab => {
        tab.classList.remove('active');
    });

    if (event && event.target) {
        const clickedLi = event.target.closest('li');
        if (clickedLi) clickedLi.classList.add('active');
    }

    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });

    const targetTab = document.getElementById(tabName + '-tab');
    if (targetTab) targetTab.style.display = 'block';
}

// Toggle AI dialog
function toggleAIDialog() {
    var dialog = document.getElementById('aiDialog');
    if (dialog.style.display === 'none' || dialog.style.display === '') {
        dialog.style.display = 'flex';
    } else {
        dialog.style.display = 'none';
    }
}

// Drag & drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
}

// ------------------ Dataset handling ------------------
function handleDatasetDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleDatasetFiles(e.dataTransfer.files);
}

function handleDatasetSelect(e) {
    if (e.target.files.length) handleDatasetFiles(e.target.files);
}

function handleDatasetFiles(files) {
    const file = files[0];
    if (!file.name.endsWith('.zip')) {
        showNotification('Please upload a ZIP dataset file', 'error');
        return;
    }
    selectedDatasetFile = file;
    document.getElementById('datasetFileName').textContent = file.name;
    document.getElementById('datasetFileInfo').style.display = 'flex';
    showNotification(`Selected dataset: ${file.name}`, 'info');
}

// ------------------ Model file handling ------------------
function handleModelDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleModelFiles(e.dataTransfer.files);
}

function handleModelSelect(e) {
    if (e.target.files.length) handleModelFiles(e.target.files);
}

function handleModelFiles(files) {
    const file = files[0];
    if (!file.name.endsWith('.pth')) {
        showNotification('Please upload a PTH model file', 'error');
        return;
    }
    selectedModelFile = file;
    document.getElementById('modelFileName').textContent = file.name;
    document.getElementById('modelFileInfo').style.display = 'flex';
    showNotification(`Selected model: ${file.name}`, 'info');
}

// ------------------ Prediction data file handling ------------------
function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleDataFiles(e.dataTransfer.files, 'dataFile', 'fileName', 'fileInfo');
}

function handleFileSelect(e) {
    if (e.target.files.length) handleDataFiles(e.target.files, 'dataFile', 'fileName', 'fileInfo');
}

// ------------------ Visualization file handling ------------------
function handleVisFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleDataFiles(e.dataTransfer.files, 'visDataFile', 'visFileName', 'visFileInfo');
}

function handleVisFileSelect(e) {
    if (e.target.files.length) handleDataFiles(e.target.files, 'visDataFile', 'visFileName', 'visFileInfo');
}

function handleDataFiles(files, fileId, nameId, infoId) {
    const file = files[0];
    const validExtensions = ['.csv', '.xlsx', '.xls', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showNotification('Unsupported format. Please select CSV, Excel, or TXT file', 'error');
        return;
    }

    if (fileId === 'dataFile') selectedDataFile = file;
    else selectedVisFile = file;

    document.getElementById(nameId).textContent = file.name;
    document.getElementById(infoId).style.display = 'flex';
    showNotification(`Selected file: ${file.name}`, 'info');
}

// Update training log
function updateTrainingLog(message) {
    const logElement = document.getElementById('trainingLog');
    logElement.innerHTML += message + '<br>';
    logElement.scrollTop = logElement.scrollHeight;
}

// ------------------ Training logic ------------------
function startTraining() {
    if (!selectedDatasetFile) {
        showNotification("Please select a dataset file first!", 'error');
        return;
    }

    const modelPath = document.getElementById('modelPath').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);

    updateTrainingLog("Uploading dataset...");

    const formData = new FormData();
    formData.append('file', selectedDatasetFile);

    fetch('/api/upload_dataset', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateTrainingLog(`Dataset uploaded: ${data.dataset_path}`);

            const trainData = {
                data_dir: data.dataset_path,
                model_path: modelPath,
                epochs: epochs,
                batch_size: batchSize
            };

            updateTrainingLog("Starting model training...");

            fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainData)
            })
            .then(response => response.json())
            .then(trainResult => {
                if (trainResult.success) {
                    updateTrainingLog("Training task started...");
                    pollTrainingStatus();
                } else {
                    updateTrainingLog(`Training failed to start: ${trainResult.error}`);
                    showNotification(`Training failed to start: ${trainResult.error}`, 'error');
                }
            });
        } else {
            updateTrainingLog(`Dataset upload failed: ${data.error}`);
            showNotification(`Dataset upload failed: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        updateTrainingLog(`Error: ${error.message}`);
        showNotification(`Error: ${error.message}`, 'error');
    });
}

function pollTrainingStatus() {
    const pollInterval = setInterval(() => {
        fetch('/api/train_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('trainingProgressBar').style.width = data.progress + '%';
                document.getElementById('progressText').textContent = data.message;

                if (data.logs && data.logs.length > 0) {
                    const currentLog = document.getElementById('trainingLog').innerText;
                    data.logs.forEach(log => {
                        if (!currentLog.includes(log)) updateTrainingLog(log);
                    });
                }

                if (data.chart_data) {
                    trainingData = data.chart_data;
                    updateTrainingCharts();
                }

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    showNotification('Model training completed!', 'success');
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    showNotification('Training error, check logs', 'error');
                }
            }
        });
    }, 1000);
}

function initTrainingCharts() {
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    const accCtx = document.getElementById('accuracyChart').getContext('2d');

    if (lossChart) lossChart.destroy();
    if (accuracyChart) accuracyChart.destroy();

    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy',
                data: [],
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { min: 0, max: 1 } }
        }
    });
}

function updateTrainingCharts() {
    if (lossChart && accuracyChart) {
        lossChart.data.labels = trainingData.epochs;
        lossChart.data.datasets[0].data = trainingData.losses;
        lossChart.update();

        accuracyChart.data.labels = trainingData.epochs;
        accuracyChart.data.datasets[0].data = trainingData.accuracies;
        accuracyChart.update();
    }
}

// ------------------ Prediction logic ------------------
function startPrediction() {
    if (!selectedModelFile || !selectedDataFile) {
        showNotification("Please select model and data files first!", 'error');
        return;
    }

    updateTrainingLog(`Uploading model file...`);
    const modelFormData = new FormData();
    modelFormData.append('file', selectedModelFile);

    fetch('/api/upload_model', { method: 'POST', body: modelFormData })
    .then(response => response.json())
    .then(modelData => {
        if (modelData.success) {
            updateTrainingLog(`Model loaded, starting prediction...`);
            const dataFormData = new FormData();
            dataFormData.append('file', selectedDataFile);

            fetch('/api/predict', { method: 'POST', body: dataFormData })
            .then(response => response.json())
            .then(predictionResult => {
                if (predictionResult.success) {
                    document.getElementById('predictionResult').textContent = predictionResult.predicted_class;
                    document.getElementById('confidenceResult').textContent = (predictionResult.confidence * 100).toFixed(2) + '%';

                    updateProbabilityChart(predictionResult.all_probabilities);
                    generateAIReport(
                        predictionResult.predicted_class,
                        predictionResult.confidence,
                        predictionResult.all_probabilities
                    );

                    updateTrainingLog(`✅ Prediction complete: ${predictionResult.predicted_class}`);
                    showNotification('Prediction completed!', 'success');
                } else {
                    updateTrainingLog(`Prediction failed: ${predictionResult.error}`);
                    showNotification(`Prediction failed: ${predictionResult.error}`, 'error');
                }
            });

        } else {
            updateTrainingLog(`Model load failed: ${modelData.error}`);
            showNotification(`Model load failed: ${modelData.error}`, 'error');
        }
    })
    .catch(error => {
        updateTrainingLog(`Error: ${error.message}`);
        showNotification(`Error: ${error.message}`, 'error');
    });
}

function updateProbabilityChart(predictions) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    if (probabilityChart) probabilityChart.destroy();

    const labels = Object.keys(predictions);
    const data = Object.values(predictions).map(p => (p * 100).toFixed(2));

    probabilityChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#e74c3c'],
                borderWidth: 2
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

// ------------------ Data Visualization (real backend) ------------------
function visualizeData() {
    if (!selectedVisFile) {
        showNotification("Please select a data file first!", 'error');
        return;
    }

    updateTrainingLog(`Visualizing data: ${selectedVisFile.name}`);

    const formData = new FormData();
    formData.append('file', selectedVisFile);

    fetch('/api/upload_for_visualization', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(uploadRes => {
        if(!uploadRes.success) throw new Error(uploadRes.error);

        return fetch('/api/visualize_split_data', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_path: uploadRes.path })
        });
    })
    .then(r => r.json())
    .then(res => {
        if(res.success) {
            const ctx = document.getElementById('waveformChart').getContext('2d');
            if(waveformChart) waveformChart.destroy();

            waveformChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: res.labels,
                    datasets: [{
                        label: 'Signal Value',
                        data: res.data,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Sample Points' } },
                        y: { title: { display: true, text: 'Value' } }
                    },
                    plugins: {
                        title: { display: true, text: `File: ${selectedVisFile.name}` },
                        zoom: {
                            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                            pan: { enabled: true, mode: 'x' }
                        }
                    }
                }
            });
            showNotification('Visualization chart updated', 'success');
            updateTrainingLog('✅ Visualization complete');
        } else {
            throw new Error(res.error);
        }
    })
    .catch(e => {
        showNotification(`Visualization failed: ${e.message}`, 'error');
        updateTrainingLog(`Visualization error: ${e.message}`);
    });
}

// ------------------ Data Split ------------------
function handleSplitFileSelect(e) {
    if (e.target.files.length) handleSplitFiles(e.target.files);
}

function handleSplitFiles(files) {
    const file = files[0];
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showNotification('Please upload a CSV or Excel file', 'error');
        return;
    }
    selectedSplitFile = file;
    document.getElementById('splitFileName').textContent = file.name;
    document.getElementById('splitFileInfo').style.display = 'flex';
    showNotification(`Selected file: ${file.name}`, 'info');
    loadSplitVisualizationData();
}

function loadSplitVisualizationData() {
    if (!selectedSplitFile) {
        showNotification('Please select a file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedSplitFile);

    fetch('/api/upload_for_visualization', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const visualizationData = {
                file_path: data.path,
                x_axis_column: 1,
                y_axis_column: 2
            };

            fetch('/api/visualize_split_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(visualizationData)
            })
            .then(response => response.json())
            .then(vizData => {
                if (vizData.success) {
                    renderVisualizationChart(vizData.labels, vizData.data, vizData.x_axis_label, vizData.y_axis_label);
                    showNotification('Waveform data loaded', 'success');

                } else {
                    showNotification(`Failed to load visualization: ${vizData.error}`, 'error');
                }
            });
        } else {
            showNotification(`File upload failed: ${data.error}`, 'error');
        }
    })
    .catch(error => showNotification(`Upload error: ${error.message}`, 'error'));
}

function renderVisualizationChart(labels, data, xAxisLabel, yAxisLabel) {
    const ctx = document.getElementById('splitVisualizationChart').getContext('2d');
    if (splitVisualizationChart) splitVisualizationChart.destroy();

    splitVisualizationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: yAxisLabel,
                data: data,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                zoom: {
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                    pan: { enabled: true, mode: 'x' }
                }
            }
        }
    });
}

// Toggle split mode (manual/equal)
function toggleSplitMode() {
    const mode = document.querySelector('input[name="splitMode"]:checked').value;
    document.getElementById('split-manual-section').style.display = mode === 'manual' ? 'block' : 'none';
    document.getElementById('split-equal-section').style.display = mode === 'equal' ? 'block' : 'none';
}

function confirmSegmentCount() {
    const segmentCount = parseInt(document.getElementById('segmentCount').value);
    if (isNaN(segmentCount) || segmentCount <= 0) return;

    const tbody = document.getElementById('splitParamsTableBody');
    tbody.innerHTML = '';
    for (let i = 0; i < segmentCount; i++) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${i + 1}</td>
            <td><input type="number" id="start_${i}" step="any" value="${i * 10.0}"></td>
            <td><input type="number" id="end_${i}" step="any" value="${(i + 1) * 10.0}"></td>
            <td><input type="text" id="name_${i}" placeholder="e.g. angle_30_state0" value="segment_${i + 1}">
        `;
        tbody.appendChild(row);
    }
    document.getElementById('splitParamsSection').style.display = 'block';
}

function previewSplit() {
    if (!selectedSplitFile) return showNotification('Please select a file first', 'error');

    const segmentCount = parseInt(document.getElementById('segmentCount').value);
    const params = [];

    for (let i = 0; i < segmentCount; i++) {
        const start = parseFloat(document.getElementById(`start_${i}`).value);
        const end = parseFloat(document.getElementById(`end_${i}`).value);
        const name = document.getElementById(`name_${i}`).value.trim();
        if (isNaN(start) || isNaN(end) || !name) return showNotification(`Please check row ${i + 1} parameters`, 'error');
        params.push({ start, end, name });
    }

    const formData = new FormData();
    formData.append('file', selectedSplitFile);
    formData.append('params', JSON.stringify(params));

    fetch('/api/preview_split', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
        if (data.success) showNotification('Preview successful, data format correct', 'success');
        else showNotification(`Preview failed: ${data.error}`, 'error');
    });
}

function executeSplit() {
    if (!selectedSplitFile) {
        return showNotification('Please select a file to split', 'error');
    }

    const formData = new FormData();
    formData.append('file', selectedSplitFile);

    const mode = document.querySelector('input[name="splitMode"]:checked').value;
    formData.append('split_mode', mode);

    if (mode === 'equal') {
        const numSegments = document.getElementById('numSegments').value;
        const namePrefix = document.getElementById('namePrefix').value;
        if (!numSegments || parseInt(numSegments) <= 0) {
            return showNotification('In equal mode, segment count must be a positive integer', 'error');
        }
        if (!namePrefix) {
            return showNotification('In equal mode, file name prefix cannot be empty', 'error');
        }
        formData.append('num_segments', numSegments);
        formData.append('name_prefix', namePrefix);
    } else {
        const segmentCount = parseInt(document.getElementById('segmentCount').value);
        if (isNaN(segmentCount) || segmentCount <= 0) {
            return showNotification('In manual mode, please confirm a valid segment count', 'error');
        }
        const params = [];
        for (let i = 0; i < segmentCount; i++) {
            const start = parseFloat(document.getElementById(`start_${i}`).value);
            const end = parseFloat(document.getElementById(`end_${i}`).value);
            const name = document.getElementById(`name_${i}`).value.trim();
            if (isNaN(start) || isNaN(end) || !name) {
                return showNotification(`Please check manual split row ${i + 1} parameters`, 'error');
            }
            if (start >= end) {
                return showNotification(`Row ${i + 1}: start point must be less than end point`, 'error');
            }
            params.push({ start, end, name });
        }
        formData.append('params', JSON.stringify(params));
    }

    showNotification('Executing split...', 'info');

    fetch('/api/split_signal', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showNotification(`Split complete! Generated ${data.file_count} files, downloading...`, 'success');
            if (data.download_url) {
                const link = document.createElement('a');

                link.href = data.download_url;
                link.download = data.download_url.split('/').pop();
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        } else {
            showNotification(`Split failed: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        showNotification(`Split error: ${error.message}`, 'error');
    });
}

// ------------------ QTBFS Scoring ------------------
let state0WaveformChart = null;
let currentWaveformChart = null;
let qtbfsRadarChart = null;

function handleQtbfsFileUpload(type, event) {
    const fileInput = event.target;
    const files = fileInput.files;
    if (files.length === 0) return;

    if (type === 'state0') {
        state0Files = Array.from(files);
    } else {
        currentFiles = Array.from(files);
    }

    const fileListDiv = document.getElementById(type === 'state0' ? 'state0FileList' : 'currentFileList');
    fileListDiv.innerHTML = '';
    for (const file of files) {
        const fileElement = document.createElement('div');
        fileElement.className = 'file-item';
        fileElement.textContent = file.name;
        fileListDiv.appendChild(fileElement);
    }

    const lastFile = files[files.length - 1];
    previewQtbfsWaveform(lastFile, type);
    showNotification(`Selected ${files.length} ${type === 'state0' ? 'reference' : 'current'} file(s)`, 'info');
}

function previewQtbfsWaveform(file, type) {
    const formData = new FormData();
    formData.append('file', file);

    showNotification(`Generating waveform preview for ${file.name}...`, 'info');

    fetch('/api/preview_waveform', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            renderQtbfsWaveformChart(result.data, type, file.name);
            showNotification('Waveform preview generated', 'success');
        } else {
            showNotification(`Preview failed: ${result.error}`, 'error');
        }
    })
    .catch(error => {
        showNotification(`Preview request failed: ${error.message}`, 'error');
    });
}

function renderQtbfsWaveformChart(chartData, type, fileName) {
    const canvasId = type === 'state0' ? 'state0WaveformChart' : 'currentWaveformChart';
    const placeholderId = type === 'state0' ? 'state0Placeholder' : 'currentPlaceholder';
    const chartVar = type === 'state0' ? 'state0WaveformChart' : 'currentWaveformChart';

    const ctx = document.getElementById(canvasId).getContext('2d');

    if (window[chartVar]) {
        window[chartVar].destroy();
    }

    document.getElementById(placeholderId).style.display = 'none';

    window[chartVar] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels || Array.from({length: chartData.raw.length}, (_, i) => i),
            datasets: [
                {
                    label: 'Raw Signal',
                    data: chartData.raw,
                    borderColor: '#bdc3c7',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Processed Signal (Baseline Corrected)',
                    data: chartData.processed,
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: fileName },
                legend: { position: 'bottom' },
                zoom: {
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                    pan: { enabled: true, mode: 'x' }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Time (s)' } },
                y: { title: { display: true, text: 'Resistance (Ω)' } }
            }
        }
    });
}

function calculateQTBFSScore() {
    if (currentFiles.length === 0) return showNotification('Please upload current state files', 'error');
    if (state0Files.length === 0) return showNotification('Please upload state0 reference files', 'error');

    const formData = new FormData();
    currentFiles.forEach(file => formData.append('current_files', file));
    state0Files.forEach(file => formData.append('state0_files', file));

    showNotification('Calculating QTBFS score...', 'info');

    fetch('/api/qtbfs_calculate', { method: 'POST', body: formData })
    .then(async response => {
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText.substring(0, 200)}...`);
        }
        return response.json();
    })
    .then(res => {
        if (res.success) {
            displayQTBFSResult(res.result);
            showNotification('Score calculation complete', 'success');
        } else {
            showNotification(`Score calculation failed: ${res.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('QTBFS calculation error:', error);
        showNotification(`Request failed: ${error.message}`, 'error');
    });
}

function displayQTBFSResult(res) {
    const resultsDiv = document.getElementById('qtbfsResults');
    const radarDiv = document.getElementById('qtbfsRadarContainer');

    resultsDiv.style.display = 'block';
    radarDiv.style.display = 'block';

    const scoreColor = res.total_score > 85 ? '#2ecc71' : (res.total_score > 50 ? '#f1c40f' : '#e74c3c');

    resultsDiv.innerHTML = `
        <h3><i class="fas fa-poll-h"></i> Score Overview</h3>
        <div class="result-grid">
            <div class="result-card main-score">
                <div class="score-circle" style="--score-color:${scoreColor};">
                    ${res.total_score}
                </div>
                <p class="stage-text">${res.stage}</p>
            </div>
            <div class="result-card">
                <h4>Domain I: Mechanical Load</h4>
                <div class="domain-score">${res.domain_I.total.toFixed(1)} / 40</div>
            </div>
            <div class="result-card">
                <h4>Domain II: Dynamic Adaptation</h4>
                <div class="domain-score">${res.domain_II.total.toFixed(1)} / 35</div>
            </div>
            <div class="result-card">
                <h4>Domain III: Functional Reserve</h4>
                <div class="domain-score">${res.domain_III.total.toFixed(1)} / 25</div>
            </div>
        </div>
    `;

    renderQtbfsRadarChart(res.domain_I.total, res.domain_II.total, res.domain_III.total);
}

function renderQtbfsRadarChart(d1, d2, d3) {
    const ctx = document.getElementById('qtbfsRadarChart').getContext('2d');
    if (qtbfsRadarChart) {
        qtbfsRadarChart.destroy();
    }
    qtbfsRadarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Domain I: Mechanical Load (40)', 'Domain II: Dynamic Adaptation (35)', 'Domain III: Functional Reserve (25)'],
            datasets: [{
                label: 'Rehab Score',
                data: [d1, d2, d3],

                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgb(52, 152, 219)',
                pointBackgroundColor: 'rgb(52, 152, 219)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(52, 152, 219)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { display: true },
                    suggestedMin: 0,
                    suggestedMax: 30
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
}

// ------------------ AI Report ------------------
function generateAIReport(predictedClass, confidence, probabilities) {
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const recommendations = [
        "Continue the current rehab training plan, maintaining 3-4 sessions per week.",
        "Based on current angle performance, consider increasing training intensity and duration for this angle.",
        "Stop immediately if discomfort occurs during training and consult your physician.",
        "Consider combining with physical therapy for accelerated recovery."
    ];

    let reportHTML = `
        <div style="padding: 1.5rem;">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; font-size: 1.4rem;">
                <i class="fas fa-file-medical-alt"></i> Rehab Assessment Report
            </h3>
            <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <strong>🎯 Primary Angle:</strong> <span style="color: #3498db; font-weight: bold;">${predictedClass}</span>
                </p>
                <p style="font-size: 1.2rem;">
                    <strong>📊 Match Confidence:</strong> <span style="color: #2ecc71; font-weight: bold;">${(confidence * 100).toFixed(2)}%</span>
                </p>
            </div>

            <h4 style="color: #3498db; margin: 1.5rem 0 1rem; font-size: 1.2rem;">Angle Component Analysis</h4>
            <div style="margin-bottom: 1.5rem;">`;

    sortedProbs.forEach(([angle, prob], index) => {
        const marker = index === 0 ? '🏆' : '▫️';
        reportHTML += `<p style="margin-bottom: 0.5rem;">${marker} ${angle}: ${(prob * 100).toFixed(2)}%</p>`;
    });

    reportHTML += `</div><h4 style="color: #3498db; margin: 1.5rem 0 1rem; font-size: 1.2rem;">Rehab Recommendations</h4><ul>`;
    recommendations.slice(0, 3).forEach(rec => reportHTML += `<li style="margin-bottom: 0.5rem;">${rec}</li>`);
    reportHTML += `</ul></div>`;

    document.getElementById('aiReportContent').innerHTML = reportHTML;
    switchTab('analysis');
}

// ------------------ AI Dialog ------------------
function askAI() {
    const question = document.getElementById('aiQuestionInput').value;
    if (!question.trim()) return;

    const contentDiv = document.getElementById('aiDialogContent');
    const userMessage = document.createElement('div');
    userMessage.className = 'doctor-message';
    userMessage.innerHTML = `<div class="message">${question}</div><div class="doctor-avatar">R</div>`;
    contentDiv.appendChild(userMessage);

    setTimeout(() => {
        const aiMessage = document.createElement('div');
        aiMessage.className = 'ai-message';
        aiMessage.innerHTML = `<div class="ai-avatar">AI</div><div class="message">Thank you for your question. Please refer to the relevant technical documentation.</div>`;
        contentDiv.appendChild(aiMessage);
        contentDiv.scrollTop = contentDiv.scrollHeight;
    }, 1000);

    document.getElementById('aiQuestionInput').value = '';
    contentDiv.scrollTop = contentDiv.scrollHeight;
}

document.addEventListener('DOMContentLoaded', function() {
    initTrainingCharts();
    document.getElementById('aiQuestionInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') askAI();
    });

    const probCtx = document.getElementById('probabilityChart').getContext('2d');
    probabilityChart = new Chart(probCtx, { type: 'pie', data: { datasets: [{ data: [] }] } });

    const waveCtx = document.getElementById('waveformChart').getContext('2d');
    waveformChart = new Chart(waveCtx, { type: 'line', data: { datasets: [] } });
});
