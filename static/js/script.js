// 全局变量
let probabilityChart = null;
let waveformChart = null;
let lossChart = null;
let accuracyChart = null;
let splitVisualizationChart = null;

// 新增：QTBFS 可视化相关全局变量
let qtbfsVisualData = null; 
let qtbfsChart = null;

// 训练数据
let trainingData = {
    epochs: [],
    losses: [],
    accuracies: []
};

// 文件引用
let selectedDatasetFile = null;
let selectedModelFile = null;
let selectedDataFile = null;
let selectedVisFile = null;
let selectedSplitFile = null;
let state0Files = [];
let currentFiles = [];

// 显示通知
function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = 'notification ' + type;
    notification.classList.add('show');
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// 切换标签页
function switchTab(tabName) {
    document.querySelectorAll('.side-nav li').forEach(tab => {
        tab.classList.remove('active');
    });
    
    if (event && event.target) {
        // 处理点击图标或文字的情况，找到最近的 li 元素
        const clickedLi = event.target.closest('li');
        if (clickedLi) clickedLi.classList.add('active');
    }
    
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    
    const targetTab = document.getElementById(tabName + '-tab');
    if (targetTab) targetTab.style.display = 'block';
}

// 切换AI助手对话框
function toggleAIDialog() {
    var dialog = document.getElementById('aiDialog');
    if (dialog.style.display === 'none' || dialog.style.display === '') {
        dialog.style.display = 'flex';
    } else {
        dialog.style.display = 'none';
    }
}

// 文件拖拽通用处理
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

// ------------------ 数据集处理 ------------------
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
        showNotification('请上传ZIP格式的数据集文件', 'error');
        return;
    }
    selectedDatasetFile = file;
    document.getElementById('datasetFileName').textContent = file.name;
    document.getElementById('datasetFileInfo').style.display = 'flex';
    showNotification(`已选择数据集: ${file.name}`, 'info');
}

// ------------------ 模型文件处理 ------------------
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
        showNotification('请上传PTH格式的模型文件', 'error');
        return;
    }
    selectedModelFile = file;
    document.getElementById('modelFileName').textContent = file.name;
    document.getElementById('modelFileInfo').style.display = 'flex';
    showNotification(`已选择模型: ${file.name}`, 'info');
}

// ------------------ 预测数据文件处理 ------------------
function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleDataFiles(e.dataTransfer.files, 'dataFile', 'fileName', 'fileInfo');
}

function handleFileSelect(e) {
    if (e.target.files.length) handleDataFiles(e.target.files, 'dataFile', 'fileName', 'fileInfo');
}

// ------------------ 可视化文件处理 ------------------
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
        showNotification('不支持的文件格式，请选择CSV、Excel或TXT文件', 'error');
        return;
    }
    
    if (fileId === 'dataFile') selectedDataFile = file;
    else selectedVisFile = file;
    
    document.getElementById(nameId).textContent = file.name;
    document.getElementById(infoId).style.display = 'flex';
    showNotification(`已选择文件: ${file.name}`, 'info');
}

// 更新训练日志
function updateTrainingLog(message) {
    const logElement = document.getElementById('trainingLog');
    logElement.innerHTML += message + '<br>';
    logElement.scrollTop = logElement.scrollHeight;
}

// ------------------ 训练逻辑 ------------------
function startTraining() {
    if (!selectedDatasetFile) {
        showNotification("请先选择数据集文件！", 'error');
        return;
    }
    
    const modelPath = document.getElementById('modelPath').value;
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    updateTrainingLog("开始上传数据集...");
    
    const formData = new FormData();
    formData.append('file', selectedDatasetFile);
    
    fetch('/api/upload_dataset', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateTrainingLog(`数据集上传成功: ${data.dataset_path}`);
            
            const trainData = {
                data_dir: data.dataset_path,
                model_path: modelPath,
                epochs: epochs,
                batch_size: batchSize
            };
            
            updateTrainingLog("开始模型训练...");
            
            fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(trainData)
            })
            .then(response => response.json())
            .then(trainResult => {
                if (trainResult.success) {
                    updateTrainingLog("训练任务已启动...");
                    pollTrainingStatus();
                } else {
                    updateTrainingLog(`训练启动失败: ${trainResult.error}`);
                    showNotification(`训练启动失败: ${trainResult.error}`, 'error');
                }
            });
        } else {
            updateTrainingLog(`数据集上传失败: ${data.error}`);
            showNotification(`数据集上传失败: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        updateTrainingLog(`出错: ${error.message}`);
        showNotification(`出错: ${error.message}`, 'error');
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
                    showNotification('模型训练完成！', 'success');
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    showNotification('训练出错，请查看日志', 'error');
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
                label: '训练损失',
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
                label: '准确率',
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

// ------------------ 预测逻辑 ------------------
function startPrediction() {
    if (!selectedModelFile || !selectedDataFile) {
        showNotification("请先选择模型和数据文件！", 'error');
        return;
    }
    
    updateTrainingLog(`开始上传模型文件...`);
    const modelFormData = new FormData();
    modelFormData.append('file', selectedModelFile);
    
    fetch('/api/upload_model', { method: 'POST', body: modelFormData })
    .then(response => response.json())
    .then(modelData => {
        if (modelData.success) {
            updateTrainingLog(`模型加载成功，开始预测...`);
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
                    
                    updateTrainingLog(`✅ 预测完成: ${predictionResult.predicted_class}`);
                    showNotification('预测完成！', 'success');
                } else {
                    updateTrainingLog(`预测失败: ${predictionResult.error}`);
                    showNotification(`预测失败: ${predictionResult.error}`, 'error');
                }
            });
        } else {
            updateTrainingLog(`模型加载失败: ${modelData.error}`);
            showNotification(`模型加载失败: ${modelData.error}`, 'error');
        }
    })
    .catch(error => {
        updateTrainingLog(`出错: ${error.message}`);
        showNotification(`出错: ${error.message}`, 'error');
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

// ------------------ 数据可视化 (真实后端调用) ------------------
function visualizeData() {
    if (!selectedVisFile) {
        showNotification("请先选择数据文件！", 'error');
        return;
    }
    
    updateTrainingLog(`开始可视化数据: ${selectedVisFile.name}`);
    
    const formData = new FormData();
    formData.append('file', selectedVisFile);

    // 1. 上传文件
    fetch('/api/upload_for_visualization', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(uploadRes => {
        if(!uploadRes.success) throw new Error(uploadRes.error);
        
        // 2. 获取数据点
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
                        label: '信号值',
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
                        x: { title: { display: true, text: '采样点' } },
                        y: { title: { display: true, text: '数值' } }
                    },
                    plugins: {
                        title: { display: true, text: `文件: ${selectedVisFile.name}` },
                        zoom: {
                            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                            pan: { enabled: true, mode: 'x' }
                        }
                    }
                }
            });
            showNotification('可视化图表已更新', 'success');
            updateTrainingLog('✅ 可视化成功');
        } else {
            throw new Error(res.error);
        }
    })
    .catch(e => {
        showNotification(`可视化失败: ${e.message}`, 'error');
        updateTrainingLog(`可视化出错: ${e.message}`);
    });
}

// ------------------ 数据分割 ------------------
function handleSplitFileSelect(e) {
    if (e.target.files.length) handleSplitFiles(e.target.files);
}

function handleSplitFiles(files) {
    const file = files[0];
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        showNotification('请上传CSV或Excel格式的文件', 'error');
        return;
    }
    selectedSplitFile = file;
    document.getElementById('splitFileName').textContent = file.name;
    document.getElementById('splitFileInfo').style.display = 'flex';
    showNotification(`已选择文件: ${file.name}`, 'info');
    loadSplitVisualizationData(); 
}

// 加载分割可视化数据
function loadSplitVisualizationData() {
    if (!selectedSplitFile) {
        showNotification('请先选择文件', 'error');
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
                x_axis_column: 1, // time
                y_axis_column: 2  // resistance
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
                    showNotification('波形数据加载成功', 'success');
                } else {
                    showNotification(`加载可视化数据失败: ${vizData.error}`, 'error');
                }
            });
        } else {
            showNotification(`上传文件失败: ${data.error}`, 'error');
        }
    })
    .catch(error => showNotification(`上传文件时出错: ${error.message}`, 'error'));
}

// 渲染分割图表
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


// 切换分割模式（手动/等分）
function toggleSplitMode() {
    const mode = document.querySelector('input[name="splitMode"]:checked').value;
    document.getElementById('split-manual-section').style.display = mode === 'manual' ? 'block' : 'none';
    document.getElementById('split-equal-section').style.display = mode === 'equal' ? 'block' : 'none';
}

// 分割配置
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
            <td><input type="text" id="name_${i}" placeholder="如: angle_30_state0" value="segment_${i + 1}">
        `;
        tbody.appendChild(row);
    }
    document.getElementById('splitParamsSection').style.display = 'block';
}

function previewSplit() {
    if (!selectedSplitFile) return showNotification('请先选择文件', 'error');
    
    const segmentCount = parseInt(document.getElementById('segmentCount').value);
    const params = [];
    
    for (let i = 0; i < segmentCount; i++) {
        const start = parseFloat(document.getElementById(`start_${i}`).value);
        const end = parseFloat(document.getElementById(`end_${i}`).value);
        const name = document.getElementById(`name_${i}`).value.trim();
        if (isNaN(start) || isNaN(end) || !name) return showNotification(`请检查第${i + 1}行参数`, 'error');
        params.push({ start, end, name });
    }
    
    const formData = new FormData();
    formData.append('file', selectedSplitFile);
    formData.append('params', JSON.stringify(params));
    
    fetch('/api/preview_split', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
        if (data.success) showNotification('预览成功，数据格式正确', 'success');
        else showNotification(`预览失败: ${data.error}`, 'error');
    });
}

function executeSplit() {
    if (!selectedSplitFile) {
        return showNotification('请先选择要分割的文件', 'error');
    }

    const formData = new FormData();
    formData.append('file', selectedSplitFile);

    const mode = document.querySelector('input[name="splitMode"]:checked').value;
    formData.append('split_mode', mode);

    if (mode === 'equal') {
        const numSegments = document.getElementById('numSegments').value;
        const namePrefix = document.getElementById('namePrefix').value;
        if (!numSegments || parseInt(numSegments) <= 0) {
            return showNotification('等分模式下，分割段数必须为正整数', 'error');
        }
        if (!namePrefix) {
            return showNotification('等分模式下，文件名前缀不能为空', 'error');
        }
        formData.append('num_segments', numSegments);
        formData.append('name_prefix', namePrefix);
    } else { // manual mode
        const segmentCount = parseInt(document.getElementById('segmentCount').value);
        if (isNaN(segmentCount) || segmentCount <= 0) {
            return showNotification('手动模式下，请先确认有效的分割段数', 'error');
        }
        const params = [];
        for (let i = 0; i < segmentCount; i++) {
            const start = parseFloat(document.getElementById(`start_${i}`).value);
            const end = parseFloat(document.getElementById(`end_${i}`).value);
            const name = document.getElementById(`name_${i}`).value.trim();
            if (isNaN(start) || isNaN(end) || !name) {
                return showNotification(`请检查手动分割第 ${i + 1} 行的参数`, 'error');
            }
            if (start >= end) {
                return showNotification(`第 ${i + 1} 行的起始点必须小于结束点`, 'error');
            }
            params.push({ start, end, name });
        }
        formData.append('params', JSON.stringify(params));
    }

    showNotification('正在执行分割...', 'info');

    fetch('/api/split_signal', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showNotification(`分割完成！生成了 ${data.file_count} 个文件，准备下载...`, 'success');
            if (data.download_url) {
                // 创建一个隐藏的a标签来触发下载
                const link = document.createElement('a');
                link.href = data.download_url;
                // 从URL中提取文件名
                link.download = data.download_url.split('/').pop();
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        } else {
            showNotification(`分割失败: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        showNotification(`执行分割时出错: ${error.message}`, 'error');
    });
}

// ------------------ QTBFS 评分 ------------------
let state0WaveformChart = null;
let currentWaveformChart = null;
let qtbfsRadarChart = null;

function handleQtbfsFileUpload(type, event) {
    const fileInput = event.target;
    const files = fileInput.files;
    if (files.length === 0) return;

    // 更新全局文件列表
    if (type === 'state0') {
        state0Files = Array.from(files);
    } else {
        currentFiles = Array.from(files);
    }

    // 更新界面上的文件列表显示
    const fileListDiv = document.getElementById(type === 'state0' ? 'state0FileList' : 'currentFileList');
    fileListDiv.innerHTML = '';
    for (const file of files) {
        const fileElement = document.createElement('div');
        fileElement.className = 'file-item';
        fileElement.textContent = file.name;
        fileListDiv.appendChild(fileElement);
    }

    // 预览最后一个上传的文件
    const lastFile = files[files.length - 1];
    previewQtbfsWaveform(lastFile, type);
    showNotification(`已选择 ${files.length} 个${type === 'state0' ? '参考' : '当前'}文件`, 'info');
}

function previewQtbfsWaveform(file, type) {
    const formData = new FormData();
    formData.append('file', file);

    showNotification(`正在生成 ${file.name} 的波形预览...`, 'info');

    fetch('/api/preview_waveform', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            renderQtbfsWaveformChart(result.data, type, file.name);
            showNotification('波形预览生成成功', 'success');
        } else {
            showNotification(`预览失败: ${result.error}`, 'error');
        }
    })
    .catch(error => {
        showNotification(`预览请求失败: ${error.message}`, 'error');
    });
}

function renderQtbfsWaveformChart(chartData, type, fileName) {
    const canvasId = type === 'state0' ? 'state0WaveformChart' : 'currentWaveformChart';
    const placeholderId = type === 'state0' ? 'state0Placeholder' : 'currentPlaceholder';
    const chartVar = type === 'state0' ? 'state0WaveformChart' : 'currentWaveformChart';

    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // 销毁旧图表
    if (window[chartVar]) {
        window[chartVar].destroy();
    }

    // 隐藏占位符
    document.getElementById(placeholderId).style.display = 'none';

    window[chartVar] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels || Array.from({length: chartData.raw.length}, (_, i) => i),
            datasets: [
                {
                    label: '原始信号',
                    data: chartData.raw,
                    borderColor: '#bdc3c7', // 灰色
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: '处理后信号 (基线校正)',
                    data: chartData.processed,
                    borderColor: '#2ecc71', // 绿色
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
                x: { title: { display: true, text: '时间 (s)' } },
                y: { title: { display: true, text: '电阻 (Ω)' } }
            }
        }
    });
}


function calculateQTBFSScore() {
    if (currentFiles.length === 0) return showNotification('请上传当前状态文件', 'error');
    if (state0Files.length === 0) return showNotification('请上传状态0参考文件', 'error');
    
    const formData = new FormData();
    currentFiles.forEach(file => formData.append('current_files', file));
    state0Files.forEach(file => formData.append('state0_files', file));
    
    showNotification('正在计算QTBFS评分...', 'info');

    fetch('/api/qtbfs_calculate', { method: 'POST', body: formData })
    .then(async response => {
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`服务器错误 (${response.status}): ${errorText.substring(0, 200)}...`);
        }
        return response.json();
    })
    .then(res => {
        if (res.success) {
            displayQTBFSResult(res.result);
            showNotification('评分计算完成', 'success');
        } else {
            showNotification(`评分计算失败: ${res.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('计算QTBFS评分时出错:', error);
        showNotification(`请求失败: ${error.message}`, 'error');
    });
}

function displayQTBFSResult(res) {
    const resultsDiv = document.getElementById('qtbfsResults');
    const radarDiv = document.getElementById('qtbfsRadarContainer');
    
    resultsDiv.style.display = 'block';
    radarDiv.style.display = 'block';

    const scoreColor = res.total_score > 85 ? '#2ecc71' : (res.total_score > 50 ? '#f1c40f' : '#e74c3c');
    
    resultsDiv.innerHTML = `
        <h3><i class="fas fa-poll-h"></i> 评分概览</h3>
        <div class="result-grid">
            <div class="result-card main-score">
                <div class="score-circle" style="--score-color:${scoreColor};">
                    ${res.total_score}
                </div>
                <p class="stage-text">${res.stage}</p>
            </div>
            <div class="result-card">
                <h4>域I: 力学承载能力</h4>
                <div class="domain-score">${res.domain_I.total.toFixed(1)} / 40</div>
            </div>
            <div class="result-card">
                <h4>域II: 动态适应能力</h4>
                <div class="domain-score">${res.domain_II.total.toFixed(1)} / 35</div>
            </div>
            <div class="result-card">
                <h4>域III: 功能储备能力</h4>
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
            labels: ['域I:力学承载 (40)', '域II:动态适应 (35)', '域III:功能储备 (25)'],
            datasets: [{
                label: '康复得分',
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


// ------------------ AI 报告 ------------------
function generateAIReport(predictedClass, confidence, probabilities) {
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const recommendations = [
        "建议继续当前康复训练方案，保持每周3-4次的训练频率。",
        "根据当前角度表现，可适当增加该角度的训练强度和持续时间。",
        "注意训练过程中如有不适感应立即停止，并及时就医复查。",
        "建议配合物理治疗师进行手法治疗，以加速康复进程。"
    ];
    
    let reportHTML = `
        <div style="padding: 1.5rem;">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; font-size: 1.4rem;">
                <i class="fas fa-file-medical-alt"></i> 康复评估报告
            </h3>
            <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <strong>🎯 主要角度:</strong> <span style="color: #3498db; font-weight: bold;">${predictedClass}</span>
                </p>
                <p style="font-size: 1.2rem;">
                    <strong>📊 主角度匹配度:</strong> <span style="color: #2ecc71; font-weight: bold;">${(confidence * 100).toFixed(2)}%</span>
                </p>
            </div>
            <h4 style="color: #3498db; margin: 1.5rem 0 1rem; font-size: 1.2rem;">各角度成分分析</h4>
            <div style="margin-bottom: 1.5rem;">`;
    
    sortedProbs.forEach(([angle, prob], index) => {
        const marker = index === 0 ? '🏆' : '▫️';
        reportHTML += `<p style="margin-bottom: 0.5rem;">${marker} ${angle}: ${(prob * 100).toFixed(2)}%</p>`;
    });
    
    reportHTML += `</div><h4 style="color: #3498db; margin: 1.5rem 0 1rem; font-size: 1.2rem;">康复建议</h4><ul>`;
    recommendations.slice(0, 3).forEach(rec => reportHTML += `<li style="margin-bottom: 0.5rem;">${rec}</li>`);
    reportHTML += `</ul></div>`;
    
    document.getElementById('aiReportContent').innerHTML = reportHTML;
    switchTab('analysis');
}

// ------------------ AI对话 ------------------
function askAI() {
    const question = document.getElementById('aiQuestionInput').value;
    if (!question.trim()) return;
    
    const contentDiv = document.getElementById('aiDialogContent');
    const userMessage = document.createElement('div');
    userMessage.className = 'doctor-message';
    userMessage.innerHTML = `<div class="message">${question}</div><div class="doctor-avatar">研</div>`;
    contentDiv.appendChild(userMessage);
    
    setTimeout(() => {
        const aiMessage = document.createElement('div');
        aiMessage.className = 'ai-message';
        aiMessage.innerHTML = `<div class="ai-avatar">AI</div><div class="message">收到您的提问。建议您参考相关技术文档。</div>`;
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
    
    // 初始化空图表以占位
    const probCtx = document.getElementById('probabilityChart').getContext('2d');
    probabilityChart = new Chart(probCtx, { type: 'pie', data: { datasets: [{ data: [] }] } });
    
    const waveCtx = document.getElementById('waveformChart').getContext('2d');
    waveformChart = new Chart(waveCtx, { type: 'line', data: { datasets: [] } });
});