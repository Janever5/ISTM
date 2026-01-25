<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 膝关节康复波形分类系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* ... 保留原有 CSS 样式不变 ... */
        /* 为了节省篇幅，这里省略 CSS，请使用原文件的 CSS */
        /* 建议添加以下样式以支持多文件上传列表显示 */
        .file-list {
            max-height: 150px;
            overflow-y: auto;
            background: #f8f9fa;
            border: 1px solid #ddd;
            padding: 10px;
            margin-top: 5px;
            border-radius: 5px;
            font-size: 0.9em;
        }
    </style>
    <style>
        /* 全局样式重置 */
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%); color: #333; line-height: 1.6; min-height: 100vh; }
        .app-container { display: flex; flex-direction: column; min-height: 100vh; }
        .top-nav { display: flex; justify-content: space-between; align-items: center; background: linear-gradient(90deg, #1a3a6d, #2c3e50); color: white; padding: 0 2rem; height: 70px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }
        .logo h1 { font-size: 1.6rem; font-weight: 600; }
        .user-info { display: flex; align-items: center; gap: 15px; }
        .avatar { width: 45px; height: 45px; border-radius: 50%; background: linear-gradient(135deg, #3498db, #80b0ff); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.1rem; }
        .main-container { display: flex; flex: 1; }
        .side-nav { width: 240px; background: linear-gradient(to bottom, #2c3e50, #1a2636); color: white; padding: 2rem 0; display: flex; flex-direction: column; justify-content: space-between; min-height: calc(100vh - 70px); box-shadow: 3px 0 15px rgba(0, 0, 0, 0.1); }
        .side-nav ul { list-style: none; }
        .side-nav li { padding: 1.2rem 1.8rem; cursor: pointer; transition: all 0.3s; font-size: 1.1rem; display: flex; align-items: center; }
        .side-nav li:hover { background-color: rgba(52, 152, 219, 0.3); }
        .side-nav li.active { background: linear-gradient(90deg, #3498db, #2980b9); border-left: 4px solid #f1c40f; }
        .side-nav i { margin-right: 12px; width: 24px; text-align: center; font-size: 1.2rem; }
        .ai-assistant { margin: 1.5rem; padding: 1.5rem; background: rgba(26, 38, 54, 0.7); border-radius: 12px; border-left: 5px solid #3498db; backdrop-filter: blur(10px); }
        .ai-header { display: flex; align-items: center; margin-bottom: 1rem; }
        .ai-header i { margin-right: 0.8rem; color: #3498db; font-size: 1.5rem; }
        .ai-assistant h3 { font-size: 1.2rem; font-weight: 600; }
        .ai-assistant p { font-size: 0.95rem; color: #b8c2cc; margin-bottom: 1.2rem; line-height: 1.5; }
        .ai-button { background: linear-gradient(90deg, #3498db, #2980b9); color: white; border: none; padding: 0.8rem 1.2rem; border-radius: 30px; cursor: pointer; width: 100%; transition: all 0.3s; font-weight: 600; font-size: 1rem; display: flex; align-items: center; justify-content: center; gap: 8px; }
        .ai-button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
        .content { flex: 1; padding: 2rem; overflow-y: auto; }
        .content-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
        .content-header h2 { font-size: 1.8rem; font-weight: 600; color: #2c3e50; }
        .stats-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }
        .stat-card { background: white; border-radius: 15px; padding: 1.8rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); transition: transform 0.3s; border-top: 5px solid #3498db; }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-card:nth-child(2) { border-top-color: #e74c3c; }
        .stat-card:nth-child(3) { border-top-color: #f39c12; }
        .stat-card:nth-child(4) { border-top-color: #2ecc71; }
        .stat-card h3 { color: #6c757d; font-size: 1.1rem; font-weight: 600; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 8px; }
        .stat-value { font-size: 2.2rem; font-weight: 700; color: #3498db; margin-bottom: 0.3rem; }
        .stat-value.alert { color: #e74c3c; }
        .stat-value.warning { color: #f39c12; }
        .stat-value.success { color: #2ecc71; }
        .stat-label { color: #6c757d; font-size: 1rem; }
        .ai-insights { background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2.5rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); border-left: 5px solid #3498db; }
        .ai-insights-header { display: flex; align-items: center; margin-bottom: 1.5rem; }
        .ai-insights-header i { color: #3498db; margin-right: 0.8rem; font-size: 1.5rem; }
        .ai-insights-header h3 { font-size: 1.4rem; font-weight: 600; }
        .ai-content { margin-bottom: 1.5rem; }
        .ai-content p { margin-bottom: 0.8rem; line-height: 1.6; font-size: 1.05rem; }
        .ai-details-btn { background: #f0f0f0; color: #333; padding: 0.8rem 1.5rem; border-radius: 30px; border: none; cursor: pointer; transition: all 0.3s; font-weight: 600; font-size: 1rem; }
        .ai-details-btn:hover { background: #e0e0e0; transform: translateY(-2px); }
        .form-section { background: white; border-radius: 15px; padding: 2rem; margin-bottom: 2.5rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); }
        .form-section h3 { font-size: 1.4rem; font-weight: 600; margin-bottom: 1.5rem; color: #2c3e50; display: flex; align-items: center; gap: 10px; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .form-group { margin-bottom: 1.5rem; }
        .form-group label { display: block; margin-bottom: 0.8rem; font-weight: 500; color: #555; font-size: 1.05rem; }
        .form-control { width: 100%; padding: 1rem 1.2rem; border: 1px solid #ddd; border-radius: 10px; font-size: 1rem; transition: all 0.3s; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); }
        .form-control:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2); }
        .btn { padding: 1rem 1.8rem; border: none; border-radius: 10px; cursor: pointer; transition: all 0.3s; text-align: center; font-weight: 600; font-size: 1.1rem; display: inline-flex; align-items: center; justify-content: center; gap: 8px; }
        .btn-primary { background: linear-gradient(90deg, #3498db, #2980b9); color: white; box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3); }
        .btn-primary:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(52, 152, 219, 0.4); }
        .btn-success { background: linear-gradient(90deg, #2ecc71, #27ae60); color: white; box-shadow: 0 4px 10px rgba(46, 204, 113, 0.3); }
        .btn-success:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(46, 204, 113, 0.4); }
        .btn-secondary { background: linear-gradient(90deg, #95a5a6, #7f8c8d); color: white; box-shadow: 0 4px 10px rgba(149, 165, 166, 0.3); }
        .btn-secondary:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(149, 165, 166, 0.4); }
        .btn-info { background: linear-gradient(90deg, #17a2b8, #138496); color: white; box-shadow: 0 4px 10px rgba(23, 162, 184, 0.3); }
        .btn-info:hover { transform: translateY(-3px); box-shadow: 0 6px 15px rgba(23, 162, 184, 0.4); }
        .btn-danger { background: linear-gradient(90deg, #e74c3c, #c0392b); color: white; box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3); }
        .results-section { background: white; border-radius: 15px; padding: 2rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); margin-bottom: 2.5rem; }
        .results-section h3 { font-size: 1.4rem; font-weight: 600; margin-bottom: 1.5rem; color: #2c3e50; display: flex; align-items: center; gap: 10px; }
        .result-card { text-align: center; padding: 2rem; background: linear-gradient(135deg, #e3f2fd, #bbdefb); border-radius: 15px; margin-bottom: 2rem; }
        .chart-container { height: 300px; margin-top: 2rem; }
        .file-upload-area { border: 2px dashed #3498db; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0; transition: all 0.3s; background: rgba(52, 152, 219, 0.05); }
        .file-upload-area.dragover { background: rgba(52, 152, 219, 0.1); border-color: #2980b9; }
        .file-upload-area i { font-size: 3rem; color: #3498db; margin-bottom: 1rem; }
        .file-info { background: linear-gradient(135deg, #e8f4fc, #d1e7ff); padding: 1rem; border-radius: 10px; margin-top: 1rem; font-size: 0.95rem; display: flex; align-items: center; gap: 10px; }
        .notification { position: fixed; top: 20px; right: 20px; padding: 1rem 2rem; border-radius: 10px; color: white; font-weight: bold; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); z-index: 1001; transform: translateX(200%); transition: transform 0.3s ease; }
        .notification.show { transform: translateX(0); }
        .notification.success { background: linear-gradient(90deg, #2ecc71, #27ae60); }
        .notification.error { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .notification.info { background: linear-gradient(90deg, #3498db, #2980b9); }
        .training-progress { margin-top: 1rem; }
        .progress-bar-container { width: 100%; height: 20px; background-color: #ecf0f1; border-radius: 10px; overflow: hidden; margin-bottom: 0.5rem; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #3498db, #2980b9); border-radius: 10px; transition: width 0.3s ease; }
        .progress-text { text-align: center; font-size: 0.9rem; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="top-nav">
            <div class="logo"><h1>🏥 膝关节康复波形分类系统</h1></div>
            <div class="user-info"><span class="username">研究员</span><div class="avatar">研</div></div>
        </header>

        <div class="main-container">
            <nav class="side-nav">
                <ul>
                    <li class="active" onclick="switchTab('train')"><i class="fas fa-brain"></i> <span>🧠 模型训练</span></li>
                    <li onclick="switchTab('predict')"><i class="fas fa-search"></i> <span>🔍 波形预测</span></li>
                    <li onclick="switchTab('visualize')"><i class="fas fa-chart-line"></i> <span>📊 数据可视化</span></li>
                    <li onclick="switchTab('qtbfs')"><i class="fas fa-heartbeat"></i> <span>📊 康复评分</span></li>
                    <li onclick="switchTab('split')"><i class="fas fa-cut"></i> <span>✂️ 数据分割</span></li>
                </ul>
            </nav>

            <main class="content">
                
                <div id="train-tab" class="tab-content">
                    <div class="content-header"><h2>🧠 模型训练</h2></div>
                    <div class="form-section">
                        <h3><i class="fas fa-cog"></i> 训练配置</h3>
                        <div class="form-grid">
                           <div class="form-group">
                                <label for="dataDir">📂 数据目录 (ZIP压缩包)</label>
                                <div class="file-upload-area" onclick="document.getElementById('datasetFileInput').click()">
                                    <i class="fas fa-cloud-upload-alt"></i>
                                    <p>点击选择ZIP数据集</p>
                                    <input type="file" id="datasetFileInput" style="display: none;" accept=".zip" onchange="handleDatasetSelect(event)">
                                </div>
                                <div id="datasetFileInfo" class="file-info" style="display: none;">已选择: <span id="datasetFileName"></span></div>
                            </div>
                            <div class="form-group">
                                <label>🔄 训练轮数</label>
                                <input type="number" id="epochs" class="form-control" value="50">
                            </div>
                            <div class="form-group">
                                <label>📦 批处理大小</label>
                                <input type="number" id="batchSize" class="form-control" value="64">
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="startTraining()">开始训练</button>
                    </div>
                    <div class="results-section">
                        <h3><i class="fas fa-chart-line"></i> 训练进度</h3>
                         <div class="training-progress">
                            <div class="progress-bar-container"><div class="progress-bar" id="trainingProgressBar" style="width: 0%;"></div></div>
                            <div class="progress-text" id="progressText">等待开始...</div>
                        </div>
                         <div id="trainingLog" style="height:150px; overflow-y:auto; background:#eee; padding:10px; margin-top:10px;"></div>
                    </div>
                </div>

                <div id="predict-tab" class="tab-content" style="display: none;">
                    <div class="content-header"><h2>🔍 波形预测</h2></div>
                    <div class="form-section">
                        <div class="form-group">
                            <label>📄 数据文件 (CSV/Excel)</label>
                             <input type="file" id="fileInput" class="form-control" accept=".csv,.xlsx,.xls">
                        </div>
                        <div class="form-group">
                            <label>🤖 模型文件 (.pth)</label>
                             <input type="file" id="modelFileInput" class="form-control" accept=".pth">
                        </div>
                        <button class="btn btn-success" onclick="startPrediction()">开始预测</button>
                    </div>
                    <div class="results-section">
                        <h3>预测结果: <span id="predictionResult">--</span> (置信度: <span id="confidenceResult">--</span>)</h3>
                        <div class="chart-container"><canvas id="probabilityChart"></canvas></div>
                    </div>
                </div>

                <div id="visualize-tab" class="tab-content" style="display: none;">
                     <div class="content-header"><h2>📊 数据可视化</h2></div>
                     <div class="form-section">
                         <input type="file" id="visFileInput" class="form-control" accept=".csv,.xlsx">
                         <button class="btn btn-primary" onclick="visualizeData()" style="margin-top:10px;">生成图表</button>
                     </div>
                     <div class="results-section">
                         <div class="chart-container"><canvas id="waveformChart"></canvas></div>
                     </div>
                </div>
                
                <div id="qtbfs-tab" class="tab-content" style="display: none;">
                    <div class="content-header">
                        <h2>📊 QTBFS康复评分</h2>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-clipboard-list"></i> 评分数据上传</h3>
                        <div class="alert" style="background: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                            <i class="fas fa-exclamation-triangle"></i> 注意：请确保文件名包含 "angle_XX" 或 "speed_XX" 以便系统自动识别。
                        </div>
                        
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="qtbfsState0Files"><i class="fas fa-folder-open"></i> 状态0 (健康/参考) 文件组</label>
                                <input type="file" id="qtbfsState0Files" class="form-control" multiple accept=".csv">
                                <div id="state0List" class="file-list"></div>
                            </div>
                            
                            <div class="form-group">
                                <label for="qtbfsCurrentFiles"><i class="fas fa-folder-open"></i> 当前状态 (待评估) 文件组</label>
                                <input type="file" id="qtbfsCurrentFiles" class="form-control" multiple accept=".csv">
                                <div id="currentList" class="file-list"></div>
                            </div>
                        </div>
                        
                        <button class="btn btn-primary" onclick="calculateQTBFSScore()" style="margin-top: 20px; padding: 12px 20px;">
                            <i class="fas fa-calculator"></i> 计算QTBFS评分
                        </button>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-chart-bar"></i> QTBFS评分结果</h3>
                        <div class="result-card">
                            <div>总分</div>
                            <div class="stat-value" style="color: #9b59b6; font-size: 2.5rem; margin: 15px 0;" id="qtbfsTotalScore">--</div>
                            <div>康复阶段</div>
                            <div class="stat-value" style="color: #e74c3c; font-size: 1.8rem;" id="rehabStage">--</div>
                        </div>
                        <div id="qtbfsDetails" style="display:none;">
                            </div>
                    </div>
                </div>
                
                <div id="split-tab" class="tab-content" style="display: none;">
                    <div class="content-header">
                        <h2>✂️ 信号数据分割</h2>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-file-upload"></i> 数据文件上传</h3>
                        <div class="form-group">
                            <label for="splitSourceFile"><i class="fas fa-file"></i> 选择CSV或Excel文件</label>
                            <input type="file" id="splitSourceFile" class="form-control" accept=".csv,.xlsx,.xls">
                        </div>
                        
                        <div class="form-group">
                            <label><i class="fas fa-project-diagram"></i> 坐标轴选择</label>
                            <div style="display: flex; gap: 20px; margin-top: 10px;">
                                <div style="flex: 1;">
                                    <label>X轴列 (时间):</label>
                                    <select id="xAxisColumn" class="form-control">
                                        <option value="1">第2列</option>
                                        <option value="0">第1列</option>
                                        <option value="2">第3列</option>
                                    </select>
                                </div>
                                <div style="flex: 1;">
                                    <label>Y轴列 (电阻):</label>
                                    <select id="yAxisColumn" class="form-control">
                                        <option value="2">第3列</option>
                                        <option value="1">第2列</option>
                                        <option value="0">第1列</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="visualizeSplitData()">绘制波形图</button>
                    </div>
                    
                    <div class="results-section">
                        <h3><i class="fas fa-wave-square"></i> 信号波形图</h3>
                        <div class="chart-container"><canvas id="splitWaveformChart"></canvas></div>
                    </div>
                    
                    <div class="form-section">
                        <h3><i class="fas fa-edit"></i> 分割参数配置</h3>
                        <div id="splitParamsContainer"></div>
                        <button class="btn btn-success" onclick="addSplitParam()" style="margin-top:10px;"><i class="fas fa-plus"></i> 添加分割段</button>
                        
                        <div class="form-actions" style="margin-top:20px;">
                            <button class="btn btn-info" onclick="previewSplit()"><i class="fas fa-eye"></i> 预览分割</button>
                            <button class="btn btn-primary" onclick="performSignalSplit()"><i class="fas fa-download"></i> 执行并下载ZIP</button>
                        </div>
                        <div id="splitPreview" style="margin-top:10px; color:#666;"></div>
                    </div>
                </div>

            </main>
        </div>
    </div>
    
    <div class="notification" id="notification"></div>

    <script>
        // 通用功能
        function showNotification(msg, type='info') {
            const el = document.getElementById('notification');
            el.textContent = msg;
            el.className = `notification ${type} show`;
            setTimeout(() => el.classList.remove('show'), 3000);
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.getElementById(`${tabName}-tab`).style.display = 'block';
            document.querySelectorAll('.side-nav li').forEach(el => el.classList.remove('active'));
            // 简单处理 active 样式
            event.currentTarget.classList.add('active');
            
            // 如果是切换到分割或可视化tab，且图表实例存在，需要重绘
            if (tabName === 'split' && window.splitWaveformChart) window.splitWaveformChart.resize();
            if (tabName === 'visualize' && window.waveformChart) window.waveformChart.resize();
        }

        // QTBFS 文件列表显示
        document.getElementById('qtbfsState0Files').addEventListener('change', function(e) {
            const list = document.getElementById('state0List');
            list.innerHTML = `已选择 ${this.files.length} 个文件:<br>` + Array.from(this.files).map(f => f.name).join('<br>');
        });
        document.getElementById('qtbfsCurrentFiles').addEventListener('change', function(e) {
            const list = document.getElementById('currentList');
            list.innerHTML = `已选择 ${this.files.length} 个文件:<br>` + Array.from(this.files).map(f => f.name).join('<br>');
        });

        // 训练相关逻辑
        function handleDatasetSelect(e) {
            const file = e.target.files[0];
            if(file) document.getElementById('datasetFileName').textContent = file.name;
        }

        async function startTraining() {
            const fileInput = document.getElementById('datasetFileInput');
            if(!fileInput.files.length) return showNotification('请先选择ZIP数据集', 'error');
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            showNotification('正在上传数据集...', 'info');
            const uploadRes = await fetch('/api/upload_dataset', { method: 'POST', body: formData }).then(r=>r.json());
            
            if(!uploadRes.success) return showNotification(uploadRes.error, 'error');
            
            // 开始训练
            const trainRes = await fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    data_dir: uploadRes.dataset_path,
                    epochs: document.getElementById('epochs').value,
                    batch_size: document.getElementById('batchSize').value
                })
            }).then(r=>r.json());
            
            if(trainRes.success) {
                showNotification('训练已启动', 'success');
                pollTrainingStatus();
            } else {
                showNotification(trainRes.error, 'error');
            }
        }

        function pollTrainingStatus() {
            const interval = setInterval(async () => {
                const res = await fetch('/api/train_status').then(r=>r.json());
                document.getElementById('trainingProgressBar').style.width = res.progress + '%';
                document.getElementById('progressText').textContent = res.message;
                const log = document.getElementById('trainingLog');
                log.innerHTML = res.logs.join('<br>');
                log.scrollTop = log.scrollHeight;
                
                if(res.status === 'completed' || res.status === 'error') clearInterval(interval);
            }, 1000);
        }

        // 预测逻辑 (简化版)
        async function startPrediction() {
            const file = document.getElementById('fileInput').files[0];
            const model = document.getElementById('modelFileInput').files[0];
            if(!file || !model) return showNotification('请选择数据和模型文件', 'error');
            
            // 上传模型
            const modelForm = new FormData();
            modelForm.append('file', model);
            await fetch('/api/upload_model', { method: 'POST', body: modelForm });
            
            // 预测
            const dataForm = new FormData();
            dataForm.append('file', file);
            const res = await fetch('/api/predict', { method: 'POST', body: dataForm }).then(r=>r.json());
            
            if(res.success) {
                document.getElementById('predictionResult').textContent = res.predicted_class;
                document.getElementById('confidenceResult').textContent = (res.confidence*100).toFixed(2)+'%';
                // 绘制饼图
                drawPieChart(res.all_probabilities);
            } else {
                showNotification(res.error, 'error');
            }
        }

        function drawPieChart(data) {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            if(window.probChart) window.probChart.destroy();
            window.probChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: Object.keys(data),
                    datasets: [{ data: Object.values(data), backgroundColor: ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'] }]
                }
            });
        }

        // 可视化逻辑
        async function visualizeData() {
            const file = document.getElementById('visFileInput').files[0];
            if(!file) return showNotification('请选择文件', 'error');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const uploadRes = await fetch('/api/upload_for_visualization', { method: 'POST', body: formData }).then(r=>r.json());
            if(!uploadRes.success) return showNotification(uploadRes.error, 'error');
            
            const res = await fetch('/api/visualize_split_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ file_path: uploadRes.path })
            }).then(r=>r.json());
            
            if(res.success) {
                const ctx = document.getElementById('waveformChart').getContext('2d');
                if(window.waveformChart) window.waveformChart.destroy();
                window.waveformChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: res.labels,
                        datasets: [{ label: '数值', data: res.data, borderColor: '#3498db', tension: 0.1 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        }

        // QTBFS 计算逻辑 (修改后支持多文件上传)
        async function calculateQTBFSScore() {
            const state0Files = document.getElementById('qtbfsState0Files').files;
            const currentFiles = document.getElementById('qtbfsCurrentFiles').files;
            
            if(state0Files.length === 0 || currentFiles.length === 0) {
                return showNotification('请同时上传状态0和当前状态的文件组', 'error');
            }
            
            const formData = new FormData();
            for(let i=0; i<state0Files.length; i++) formData.append('state0_files', state0Files[i]);
            for(let i=0; i<currentFiles.length; i++) formData.append('current_files', currentFiles[i]);
            
            showNotification('正在上传并分析文件...', 'info');
            
            try {
                const res = await fetch('/api/qtbfs_calculate', {
                    method: 'POST',
                    body: formData
                }).then(r => r.json());
                
                if(res.success) {
                    const r = res.result;
                    document.getElementById('qtbfsTotalScore').textContent = r.total_score;
                    document.getElementById('rehabStage').textContent = r.stage;
                    
                    const detailsDiv = document.getElementById('qtbfsDetails');
                    detailsDiv.style.display = 'block';
                    detailsDiv.innerHTML = `
                        <div style="margin-top:20px; font-size:0.9rem;">
                            <p><strong>域I (力学):</strong> ${r.domain_I.total}/40 (A:${r.domain_I.subA_score} B:${r.domain_I.subB_score} C:${r.domain_I.subC_score})</p>
                            <p><strong>域II (动态):</strong> ${r.domain_II.total}/35 (A:${r.domain_II.subA_score} B:${r.domain_II.subB_score} C:${r.domain_II.subC_score})</p>
                            <p><strong>域III (储备):</strong> ${r.domain_III.total}/25 (A:${r.domain_III.subA_score} B:${r.domain_III.subB_score} C:${r.domain_III.subC_score})</p>
                        </div>
                    `;
                    showNotification('计算完成', 'success');
                } else {
                    showNotification('计算失败: ' + res.error, 'error');
                }
            } catch(e) {
                showNotification('网络错误: ' + e.message, 'error');
            }
        }

        // 分割功能逻辑
        function addSplitParam() {
            const div = document.createElement('div');
            div.className = 'split-row';
            div.style.marginBottom = '10px';
            div.innerHTML = `
                <input type="number" placeholder="开始(s)" style="width:100px; padding:5px;"> - 
                <input type="number" placeholder="结束(s)" style="width:100px; padding:5px;"> 
                <input type="text" placeholder="文件名(如 angle_30)" style="width:150px; padding:5px;">
                <button class="btn btn-danger" onclick="this.parentElement.remove()" style="padding:5px 10px;">×</button>
            `;
            document.getElementById('splitParamsContainer').appendChild(div);
        }

        async function visualizeSplitData() {
            const file = document.getElementById('splitSourceFile').files[0];
            if(!file) return showNotification('请选择文件', 'error');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const uploadRes = await fetch('/api/upload_for_visualization', { method: 'POST', body: formData }).then(r=>r.json());
            if(!uploadRes.success) return showNotification(uploadRes.error, 'error');
            
            window.splitFilePath = uploadRes.path; // 保存路径供分割使用
            
            const res = await fetch('/api/visualize_split_data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    file_path: uploadRes.path,
                    x_axis_column: parseInt(document.getElementById('xAxisColumn').value),
                    y_axis_column: parseInt(document.getElementById('yAxisColumn').value)
                })
            }).then(r=>r.json());
            
            if(res.success) {
                const ctx = document.getElementById('splitWaveformChart').getContext('2d');
                if(window.splitWaveformChart) window.splitWaveformChart.destroy();
                window.splitWaveformChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: res.labels,
                        datasets: [{ label: '信号值', data: res.data, borderColor: '#e74c3c', pointRadius: 0 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        }

        async function performSignalSplit() {
            if(!window.splitFilePath) return showNotification('请先上传并绘制波形图', 'error');
            
            const rows = document.querySelectorAll('#splitParamsContainer .split-row');
            const params = [];
            rows.forEach(row => {
                const inputs = row.querySelectorAll('input');
                if(inputs[2].value) {
                    params.push({
                        start: parseFloat(inputs[0].value),
                        end: parseFloat(inputs[1].value),
                        name: inputs[2].value
                    });
                }
            });
            
            if(params.length === 0) return showNotification('请添加分割参数', 'error');
            
            showNotification('正在分割并打包...', 'info');
            
            const res = await fetch('/api/split_signal', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    file_path: window.splitFilePath,
                    params: params
                })
            }).then(r=>r.json());
            
            if(res.success) {
                // 触发下载
                const link = document.createElement('a');
                link.href = res.download_url;
                link.download = 'split_files.zip';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showNotification('分割完成，正在下载ZIP', 'success');
            } else {
                showNotification(res.error, 'error');
            }
        }
    </script>
</body>
</html>