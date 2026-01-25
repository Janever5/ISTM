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
function handleSplitFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.files.length) handleSplitFiles(e.dataTransfer.files);
}

function handleSplitFileSelect(e) {
    if (e.target.files.length) handleSplitFiles(e.target.files);
}

function handleSplitFiles(files) {
    const file = files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showNotification('请上传CSV格式的文件', 'error');
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
                x_axis_column: parseInt(document.getElementById('xAxisColumn').value) || 1,
                y_axis_column: parseInt(document.getElementById('yAxisColumn').value) || 2
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
    .catch(error => showNotification('上传文件时出错', 'error'));
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

// 更新分割图表（下拉框回调）
function updateSplitVisualization() {
    loadSplitVisualizationData();
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
    if (!selectedSplitFile) return showNotification('请先选择文件', 'error');
    
    const segmentCount = parseInt(document.getElementById('segmentCount').value);
    const params = [];
    for (let i = 0; i < segmentCount; i++) {
        const start = parseFloat(document.getElementById(`start_${i}`).value);
        const end = parseFloat(document.getElementById(`end_${i}`).value);
        const name = document.getElementById(`name_${i}`).value.trim();
        params.push({ start, end, name });
    }
    
    const formData = new FormData();
    formData.append('file', selectedSplitFile);
    formData.append('params', JSON.stringify(params));
    
    fetch('/api/split_signal', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showNotification(`分割完成！生成了${data.file_count || '多'}个文件，准备下载`, 'success');
            if (data.download_url) {
                const link = document.createElement('a');
                link.href = data.download_url;
                link.download = '';
                link.click();
            }
        } else {
            showNotification(`分割失败: ${data.error}`, 'error');
        }
    });
}

// ------------------ QTBFS 评分 ------------------
function handleState0Drop(e) {
    e.preventDefault(); e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.items) {
        state0Files = Array.from(e.dataTransfer.items).filter(i => i.kind === 'file').map(i => i.getAsFile());
        showNotification(`已选择${state0Files.length}个状态0参考文件`, 'info');
    }
}
function handleState0Select(e) {
    state0Files = Array.from(e.target.files);
    showNotification(`已选择${state0Files.length}个状态0参考文件`, 'info');
}
function handleCurrentDrop(e) {
    e.preventDefault(); e.stopPropagation();
    e.target.closest('.file-upload-area').classList.remove('dragover');
    if (e.dataTransfer.items) {
        currentFiles = Array.from(e.dataTransfer.items).filter(i => i.kind === 'file').map(i => i.getAsFile());
        showNotification(`已选择${currentFiles.length}个当前状态文件`, 'info');
    }
}
function handleCurrentSelect(e) {
    currentFiles = Array.from(e.target.files);
    showNotification(`已选择${currentFiles.length}个当前状态文件`, 'info');
}

function calculateQTBFSScore() {
    if (currentFiles.length === 0) return showNotification('请上传当前状态文件', 'error');
    
    const formData = new FormData();
    // 修正Key名：不带索引下标，直接使用列表
    currentFiles.forEach(file => formData.append('current_files', file));
    state0Files.forEach(file => formData.append('state0_files', file));
    
    fetch('/api/qtbfs_calculate', { method: 'POST', body: formData })
    .then(async response => {
        // 先检查响应状态，如果是 500 或 404，手动抛出文本错误
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`服务器错误 (${response.status}): ${errorText.substring(0, 100)}...`);
        }
        return response.json();
    })
    .then(res => {
        if (res.success) {
            // 保存可视化数据到全局变量
            qtbfsVisualData = res.result.visualizations;
            displayQTBFSResult(res.result);
            showNotification('评分计算完成', 'success');
        } else {
            showNotification(`评分计算失败: ${res.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('计算QTBFS评分时出错:', error);
        // 现在这里会显示真正的服务器错误原因，而不是 JSON 解析错误
        showNotification(`请求失败: ${error.message}`, 'error');
    });
}

function displayQTBFSResult(res) {
    const div = document.getElementById('qtbfsResults');
    
    // 构造下拉菜单选项
    let options = '<option value="">-- 选择文件查看波形对比 --</option>';
    if (qtbfsVisualData) {
        if(qtbfsVisualData.state0) {
            Object.keys(qtbfsVisualData.state0).forEach(k => {
                options += `<option value="state0:${k}">状态0: ${k}</option>`;
            });
        }
        if(qtbfsVisualData.current) {
            Object.keys(qtbfsVisualData.current).forEach(k => {
                options += `<option value="current:${k}">当前状态: ${k}</option>`;
            });
        }
    }

    div.innerHTML = `
        <div class="result-card">
            <h3>QTBFS总分: ${res.total_score}</h3>
            <p class="stat-value" style="color:${res.total_score>85?'#2ecc71':'#f1c40f'}">${res.stage}</p>
            <div style="text-align:left; margin-top:20px;">
                <p>域I (力学): ${res.domain_I.total}/40</p>
                <p>域II (动态): ${res.domain_II.total}/35</p>
                <p>域III (储备): ${res.domain_III.total}/25</p>
            </div>
        </div>
        
        <div class="form-section" style="margin-top:20px;">
            <h3><i class="fas fa-wave-square"></i> 信号处理前后对比</h3>
            <div style="margin-bottom: 15px;">
                <select id="qtbfsFileSelect" class="form-control" onchange="updateQtbfsChart(this.value)">
                    ${options}
                </select>
            </div>
            <div class="chart-container">
                <canvas id="qtbfsChart"></canvas>
            </div>
        </div>
    `;
    
    // 初始化空图表
    updateQtbfsChart("");
}

function updateQtbfsChart(selection) {
    if(!selection) {
        // 如果没有选择或初始化，清空图表
        if(qtbfsChart) qtbfsChart.destroy();
        return;
    }

    const [type, key] = selection.split(':');
    const dataObj = qtbfsVisualData[type][key];
    
    if(!dataObj) return;

    const ctx = document.getElementById('qtbfsChart').getContext('2d');
    if (qtbfsChart) qtbfsChart.destroy();

    qtbfsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dataObj.labels || Array.from({length: dataObj.raw.length}, (_, i) => i),
            datasets: [
                {
                    label: '原始数据 (去均值)',
                    data: dataObj.raw,
                    borderColor: '#95a5a6', // 灰色
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: '最终处理数据 (去趋势+归零)',
                    data: dataObj.processed,
                    borderColor: '#2ecc71', // 绿色
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                title: { display: true, text: `文件: ${key} 处理效果对比` },
                zoom: {
                    zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                    pan: { enabled: true, mode: 'x' }
                }
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