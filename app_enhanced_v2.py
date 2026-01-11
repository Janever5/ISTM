import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from waveform_classifier import train_model, predict_waveform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.font_manager as fm

class WaveformClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏥 膝关节康复角度波形分类系统 - AI增强版")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        self.setup_styles()
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 创建标题
        self.create_header()
        
        # 创建 Notebook 控件用于选项卡
        self.notebook = ttk.Notebook(self.main_frame, style='Custom.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建训练选项卡
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="🧠 模型训练")
        self.setup_train_tab()
        
        # 创建预测选项卡
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="🔍 波形预测")
        self.setup_predict_tab()
        
        # 创建可视化选项卡
        self.visualize_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualize_frame, text="📊 数据可视化")
        self.setup_visualize_tab()
        
        # 创建AI分析选项卡
        self.ai_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ai_analysis_frame, text="🤖 AI智能分析")
        self.setup_ai_analysis_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 膝关节康复角度波形分类系统 v2.0")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=('微软雅黑', 12))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Header.TLabel', font=('微软雅黑', 20, 'bold'), foreground='#2c3e50', background='#f0f0f0')
        style.configure('Subtitle.TLabel', font=('微软雅黑', 14), foreground='#3498db', background='#f0f0f0')
        style.configure('Title.TLabel', font=('微软雅黑', 14, 'bold'), foreground='#3498db')
        style.configure('Custom.TNotebook', background='#f0f0f0')
        style.configure('Custom.TNotebook.Tab', font=('微软雅黑', 12), padding=[15, 8])
        style.map('Custom.TNotebook.Tab', background=[('selected', '#3498db')], foreground=[('selected', 'white')])
        
        # 按钮样式
        style.configure('Action.TButton', font=('微软雅黑', 12, 'bold'), padding=8)
        style.configure('Primary.TButton', background='#3498db', foreground='white')
        style.map('Primary.TButton', background=[('active', '#2980b9')])
        
        style.configure('Secondary.TButton', background='#95a5a6', foreground='white')
        style.map('Secondary.TButton', background=[('active', '#7f8c8d')])
        
        style.configure('Success.TButton', background='#2ecc71', foreground='white')
        style.map('Success.TButton', background=[('active', '#27ae60')])
        
        style.configure('Danger.TButton', background='#e74c3c', foreground='white')
        style.map('Danger.TButton', background=[('active', '#c0392b')])
        
        style.configure('Warning.TButton', background='#f39c12', foreground='white')
        style.map('Warning.TButton', background=[('active', '#d35400')])
    
    def create_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="🏥 膝关节康复角度波形分类系统", style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header_frame, text="AI增强版 - 专为科研与临床设计", style='Subtitle.TLabel')
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        version_label = ttk.Label(header_frame, text="v2.0", font=('微软雅黑', 11), foreground='#7f8c8d')
        version_label.pack(side=tk.RIGHT)
    
    def setup_train_tab(self):
        # 主训练框架
        train_main_frame = ttk.Frame(self.train_frame)
        train_main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 左侧面板 - 配置区域
        config_frame = ttk.LabelFrame(train_main_frame, text="⚙️ 训练配置", padding=15)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # 数据目录选择
        data_dir_frame = ttk.LabelFrame(config_frame, text="📂 数据源", padding=8)
        data_dir_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(data_dir_frame, text="数据目录:", font=('微软雅黑', 11)).pack(anchor=tk.W)
        self.data_dir_var = tk.StringVar()
        self.data_dir_entry = ttk.Entry(data_dir_frame, textvariable=self.data_dir_var, width=35, font=('微软雅黑', 10))
        self.data_dir_entry.pack(fill=tk.X, pady=8)
        ttk.Button(data_dir_frame, text="📁 浏览", command=self.browse_data_dir, style='Secondary.TButton').pack(fill=tk.X)
        
        # 模型保存路径
        model_path_frame = ttk.LabelFrame(config_frame, text="💾 模型保存", padding=8)
        model_path_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(model_path_frame, text="模型路径:", font=('微软雅黑', 11)).pack(anchor=tk.W)
        self.model_path_var = tk.StringVar(value="waveform_model.pth")
        self.model_path_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var, width=35, font=('微软雅黑', 10))
        self.model_path_entry.pack(fill=tk.X, pady=8)
        ttk.Button(model_path_frame, text="📁 浏览", command=self.browse_model_path, style='Secondary.TButton').pack(fill=tk.X)
        
        # 训练参数
        params_frame = ttk.LabelFrame(config_frame, text="🎛️ 训练参数", padding=8)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Epochs
        ttk.Label(params_frame, text="训练轮数:", font=('微软雅黑', 11)).pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=25, font=('微软雅黑', 10)).pack(fill=tk.X, pady=8)
        
        # Batch Size
        ttk.Label(params_frame, text="批处理大小:", font=('微软雅黑', 11)).pack(anchor=tk.W, pady=(15, 0))
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=25, font=('微软雅黑', 10)).pack(fill=tk.X, pady=8)
        
        # 学习率
        ttk.Label(params_frame, text="学习率:", font=('微软雅黑', 11)).pack(anchor=tk.W, pady=(15, 0))
        self.learning_rate_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=25, font=('微软雅黑', 10)).pack(fill=tk.X, pady=8)
        
        # 训练按钮
        self.train_button = ttk.Button(config_frame, text="🚀 开始训练", command=self.start_training, style='Primary.TButton')
        self.train_button.pack(fill=tk.X, pady=(25, 0))
        
        # 右侧面板 - 日志和进度
        right_frame = ttk.Frame(train_main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 训练状态
        status_frame = ttk.LabelFrame(right_frame, text="📈 训练状态", padding=15)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=8)
        
        self.status_text_var = tk.StringVar()
        self.status_text_var.set("等待开始训练...")
        status_label = ttk.Label(status_frame, textvariable=self.status_text_var, font=('微软雅黑', 11))
        status_label.pack()
        
        # 训练图表
        charts_frame = ttk.LabelFrame(right_frame, text="📊 训练过程图表", padding=8)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建matplotlib图表
        self.train_fig, (self.loss_ax, self.acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
        self.train_fig.patch.set_facecolor('#f0f0f0')
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, charts_frame)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化图表
        self.loss_ax.set_title("训练损失曲线", fontsize=12)
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.acc_ax.set_title("准确率曲线", fontsize=12)
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylabel("Accuracy")
        self.train_canvas.draw()
        
        # 训练日志
        log_frame = ttk.LabelFrame(right_frame, text="📋 训练日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.train_log_text = scrolledtext.ScrolledText(log_frame, height=8, state=tk.DISABLED, font=('微软雅黑', 10))
        self.train_log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_predict_tab(self):
        # 主预测框架
        predict_main_frame = ttk.Frame(self.predict_frame)
        predict_main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 左侧面板 - 预测配置
        config_frame = ttk.LabelFrame(predict_main_frame, text="⚙️ 预测配置", padding=15)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # 模型路径
        model_frame = ttk.LabelFrame(config_frame, text="🤖 模型文件", padding=8)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(model_frame, text="模型路径:", font=('微软雅黑', 11)).pack(anchor=tk.W)
        self.pred_model_path_var = tk.StringVar(value="waveform_model.pth")
        self.pred_model_path_entry = ttk.Entry(model_frame, textvariable=self.pred_model_path_var, width=35, font=('微软雅黑', 10))
        self.pred_model_path_entry.pack(fill=tk.X, pady=8)
        ttk.Button(model_frame, text="📁 浏览", command=self.browse_pred_model_path, style='Secondary.TButton').pack(fill=tk.X)
        
        # 文件路径
        file_frame = ttk.LabelFrame(config_frame, text="📄 待预测文件", padding=8)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(file_frame, text="数据文件:", font=('微软雅黑', 11)).pack(anchor=tk.W)
        self.csv_path_var = tk.StringVar()
        self.csv_path_entry = ttk.Entry(file_frame, textvariable=self.csv_path_var, width=35, font=('微软雅黑', 10))
        self.csv_path_entry.pack(fill=tk.X, pady=8)
        ttk.Button(file_frame, text="📁 浏览", command=self.browse_csv_file, style='Secondary.TButton').pack(fill=tk.X)
        
        # 预测按钮
        self.predict_button = ttk.Button(config_frame, text="🔍 开始预测", command=self.start_prediction, style='Success.TButton')
        self.predict_button.pack(fill=tk.X, pady=(25, 0))
        
        # 右侧面板 - 预测结果
        result_frame = ttk.Frame(predict_main_frame)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 主要预测结果
        main_result_frame = ttk.LabelFrame(result_frame, text="🎯 预测结果", padding=15)
        main_result_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 预测类别
        pred_class_frame = ttk.Frame(main_result_frame)
        pred_class_frame.pack(fill=tk.X, pady=8)
        ttk.Label(pred_class_frame, text="预测类别:", font=('微软雅黑', 12, 'bold')).pack(side=tk.LEFT)
        self.pred_class_var = tk.StringVar()
        pred_class_label = ttk.Label(pred_class_frame, textvariable=self.pred_class_var, font=('微软雅黑', 14, 'bold'), foreground='#3498db')
        pred_class_label.pack(side=tk.RIGHT)
        
        # 置信度
        confidence_frame = ttk.Frame(main_result_frame)
        confidence_frame.pack(fill=tk.X, pady=8)
        ttk.Label(confidence_frame, text="置信度:", font=('微软雅黑', 12, 'bold')).pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar()
        confidence_label = ttk.Label(confidence_frame, textvariable=self.confidence_var, font=('微软雅黑', 14, 'bold'), foreground='#2ecc71')
        confidence_label.pack(side=tk.RIGHT)
        
        # 概率分布饼图
        pie_chart_frame = ttk.LabelFrame(result_frame, text="📊 概率分布", padding=10)
        pie_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.pie_fig, self.pie_ax = plt.subplots(figsize=(6, 5))
        self.pie_fig.patch.set_facecolor('#f0f0f0')
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, pie_chart_frame)
        self.pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始饼图
        self.pie_ax.set_title("角度分类概率分布", fontsize=14, pad=20)
        self.pie_canvas.draw()
    
    def setup_visualize_tab(self):
        # 可视化框架
        visualize_main_frame = ttk.Frame(self.visualize_frame)
        visualize_main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 控制面板
        control_frame = ttk.LabelFrame(visualize_main_frame, text="🎛️ 控制面板", padding=15)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 文件选择
        file_select_frame = ttk.Frame(control_frame)
        file_select_frame.pack(fill=tk.X, pady=8)
        
        ttk.Label(file_select_frame, text="数据文件:", font=('微软雅黑', 11)).pack(side=tk.LEFT)
        self.vis_file_path_var = tk.StringVar()
        self.vis_file_entry = ttk.Entry(file_select_frame, textvariable=self.vis_file_path_var, width=60, font=('微软雅黑', 10))
        self.vis_file_entry.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        ttk.Button(file_select_frame, text="📁 浏览", command=self.browse_vis_file, style='Secondary.TButton').pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(file_select_frame, text="👁️ 查看", command=self.visualize_waveform, style='Primary.TButton').pack(side=tk.LEFT, padx=(8, 0))
        
        # 图表区域
        chart_frame = ttk.LabelFrame(visualize_main_frame, text="📉 波形图表", padding=8)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始图表
        self.ax.set_title("波形数据可视化", fontsize=14, pad=20)
        self.ax.set_xlabel("时间点", fontsize=12)
        self.ax.set_ylabel("电流/电阻值", fontsize=12)
        self.canvas.draw()
    
    def setup_ai_analysis_tab(self):
        # AI分析框架
        ai_main_frame = ttk.Frame(self.ai_analysis_frame)
        ai_main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # AI分析头部
        ai_header_frame = ttk.LabelFrame(ai_main_frame, text="🤖 AI智能分析报告", padding=15)
        ai_header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # AI Logo和标题
        ai_title_frame = ttk.Frame(ai_header_frame)
        ai_title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ai_logo = ttk.Label(ai_title_frame, text="🤖", font=('Arial', 24))
        ai_logo.pack(side=tk.LEFT)
        
        ai_title = ttk.Label(ai_title_frame, text="DeepLearning AI 分析引擎", font=('微软雅黑', 16, 'bold'), foreground='#3498db')
        ai_title.pack(side=tk.LEFT, padx=(10, 0))
        
        # AI分析结果区域
        ai_result_frame = ttk.LabelFrame(ai_main_frame, text="🔬 分析结果", padding=15)
        ai_result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建带有滚动条的文本区域
        ai_text_frame = ttk.Frame(ai_result_frame)
        ai_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.ai_analysis_text = scrolledtext.ScrolledText(ai_text_frame, font=('微软雅黑', 11), wrap=tk.WORD)
        ai_scrollbar = ttk.Scrollbar(ai_text_frame, orient=tk.VERTICAL, command=self.ai_analysis_text.yview)
        self.ai_analysis_text.configure(yscrollcommand=ai_scrollbar.set)
        
        self.ai_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ai_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化AI分析文本
        initial_text = """欢迎使用AI智能分析功能！

系统将基于以下维度为您提供专业的分析报告：
• 波形特征分析
• 角度匹配度评估
• 数据质量评估
• 康复进展预测
• 个性化建议

请先进行波形预测，AI将自动生成分析报告。
"""
        self.ai_analysis_text.insert(tk.END, initial_text)
        self.ai_analysis_text.config(state=tk.DISABLED)
    
    def browse_data_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)
    
    def browse_model_path(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def browse_pred_model_path(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.pred_model_path_var.set(file_path)
    
    def browse_csv_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Supported Files", "*.csv *.xlsx *.xls *.txt"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.csv_path_var.set(file_path)
    
    def browse_vis_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Supported Files", "*.csv *.xlsx *.xls *.txt"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.vis_file_path_var.set(file_path)
    
    def visualize_waveform(self):
        file_path = self.vis_file_path_var.get()
        if not file_path:
            messagebox.showerror("错误", "请选择一个数据文件")
            return
        
        try:
            # 读取文件
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_extension == '.xls':
                df = pd.read_excel(file_path, engine='xlrd')
            elif file_extension == '.txt':
                df = pd.read_csv(file_path, sep='\s+')
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            # 查找电流列
            columns = [col.lower() for col in df.columns]
            current_col = None
            for col in columns:
                if 'current' in col or '电流' in col or 'r' in col or '电阻' in col or 'resistance' in col:
                    current_col = df.columns[columns.index(col)]
                    break
            
            if current_col is None:
                raise ValueError(f"未找到电流或电阻列，可用列: {df.columns.tolist()}")
            
            current = df[current_col].values
            
            # 绘制图表
            self.ax.clear()
            self.ax.plot(current, linewidth=2, color='#3498db')
            self.ax.set_title(f"波形数据可视化 - {Path(file_path).name}", fontsize=14, pad=20)
            self.ax.set_xlabel("时间点", fontsize=12)
            self.ax.set_ylabel("电流/电阻值", fontsize=12)
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            
            self.status_var.set(f"已加载文件: {Path(file_path).name}")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载文件: {str(e)}")
    
    def start_training(self):
        if not self.data_dir_var.get():
            messagebox.showerror("错误", "请选择数据目录")
            return
        
        self.train_button.config(state=tk.DISABLED)
        self.status_var.set("正在训练模型...")
        self.status_text_var.set("开始训练...")
        self.progress_var.set(0)
        
        # 清空图表
        self.loss_ax.clear()
        self.acc_ax.clear()
        self.loss_ax.set_title("训练损失曲线", fontsize=12)
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.acc_ax.set_title("准确率曲线", fontsize=12)
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylabel("Accuracy")
        self.train_canvas.draw()
        
        # 在后台线程中运行训练
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def run_training(self):
        try:
            self.root.after(0, lambda: self.append_log("开始训练模型...\n"))
            
            # 训练过程中收集损失和准确率数据
            losses = []
            accuracies = []
            
            def progress_callback(epoch, total_epochs, loss, accuracy):
                # 更新进度条
                progress = (epoch + 1) / total_epochs * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_text_var.set(f"训练中... Epoch: {epoch+1}/{total_epochs}"))
                
                # 收集数据用于绘图
                losses.append(loss)
                accuracies.append(accuracy)
                
                # 更新图表
                self.root.after(0, lambda: self.update_training_charts(losses, accuracies))
                
                # 更新日志
                self.root.after(0, lambda: self.append_log(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n"))
            
            train_model(
                data_dir=self.data_dir_var.get(),
                model_save_path=self.model_path_var.get(),
                epochs=int(self.epochs_var.get()),
                batch_size=int(self.batch_size_var.get()),
                progress_callback=progress_callback
            )
            
            self.root.after(0, lambda: self.append_log("✅ 训练完成!\n"))
            self.root.after(0, lambda: messagebox.showinfo("成功", "模型训练完成!"))
            self.root.after(0, lambda: self.status_text_var.set("训练完成"))
            self.root.after(0, lambda: self.progress_var.set(100))
        except Exception as e:
            self.root.after(0, lambda e=e: self.append_log(f"❌ 训练出错: {str(e)}\n"))
            self.root.after(0, lambda e=e: messagebox.showerror("错误", f"训练过程中出现错误:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("就绪"))
    
    def update_training_charts(self, losses, accuracies):
        """更新训练过程图表"""
        # 更新损失曲线
        self.loss_ax.clear()
        self.loss_ax.plot(range(1, len(losses)+1), losses, 'b-', linewidth=2)
        self.loss_ax.set_title("训练损失曲线", fontsize=12)
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True, alpha=0.3)
        
        # 更新准确率曲线
        self.acc_ax.clear()
        self.acc_ax.plot(range(1, len(accuracies)+1), accuracies, 'g-', linewidth=2)
        self.acc_ax.set_title("准确率曲线", fontsize=12)
        self.acc_ax.set_xlabel("Epoch")
        self.acc_ax.set_ylabel("Accuracy")
        self.acc_ax.grid(True, alpha=0.3)
        
        self.train_canvas.draw()
    
    def append_log(self, message):
        self.train_log_text.config(state=tk.NORMAL)
        self.train_log_text.insert(tk.END, message)
        self.train_log_text.config(state=tk.DISABLED)
        self.train_log_text.see(tk.END)
    
    def start_prediction(self):
        if not self.pred_model_path_var.get():
            messagebox.showerror("错误", "请选择模型文件")
            return
        
        if not self.csv_path_var.get():
            messagebox.showerror("错误", "请选择数据文件")
            return
        
        self.predict_button.config(state=tk.DISABLED)
        self.status_var.set("正在进行预测...")
        
        # 在后台线程中运行预测
        prediction_thread = threading.Thread(target=self.run_prediction)
        prediction_thread.daemon = True
        prediction_thread.start()
    
    def run_prediction(self):
        try:
            result = predict_waveform(
                file_path=self.csv_path_var.get(),
                model_path=self.pred_model_path_var.get()
            )
            
            # 更新界面
            self.root.after(0, lambda: self.pred_class_var.set(result["predicted_class"]))
            self.root.after(0, lambda: self.confidence_var.set(f"{result['confidence']:.4f}"))
            
            # 更新饼图
            self.root.after(0, lambda: self.update_pie_chart(result["all_probabilities"]))
            
            # 更新AI分析
            self.root.after(0, lambda: self.update_ai_analysis(result))
            
            self.root.after(0, lambda: messagebox.showinfo("成功", "预测完成!"))
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("错误", f"预测过程中出现错误:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("就绪"))
    
    def update_pie_chart(self, probabilities):
        """更新饼图显示"""
        self.pie_ax.clear()
        
        # 按概率排序
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        sizes = [item[1] for item in sorted_items]
        
        # 生成颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # 绘制饼图
        wedges, texts, autotexts = self.pie_ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, textprops={'fontsize': 10}
        )
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        self.pie_ax.set_title("角度分类概率分布", fontsize=14, pad=20)
        self.pie_canvas.draw()
    
    def update_ai_analysis(self, result):
        """更新AI分析报告"""
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["all_probabilities"]
        
        # 按概率排序
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        analysis_text = f"""=== AI智能分析报告 ===

📅 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 核心结论:
   最佳匹配角度: {predicted_class}
   匹配置信度: {confidence:.2%}
   
📊 详细概率分布:
"""
        for i, (angle, prob) in enumerate(sorted_probs, 1):
            marker = "🏆" if i == 1 else ("⭐" if i <= 3 else "▫️")
            analysis_text += f"   {marker} {angle}: {prob:.2%}\n"
        
        # 添加专业分析
        analysis_text += f"""
🧠 AI专业解读:
   1. 当前波形与{predicted_class}角度的匹配度为{confidence:.2%}，{
    '高度匹配' if confidence > 0.8 else '较好匹配' if confidence > 0.6 else '一般匹配' if confidence > 0.4 else '较低匹配'
}。
   2. {
    '系统高度确信该波形属于此角度类别。' if confidence > 0.8 
    else '系统较确信该波形属于此角度类别，但仍需注意其他可能性。' if confidence > 0.6 
    else '存在一定的不确定性，建议结合临床经验进行判断。' if confidence > 0.4 
    else '匹配度较低，建议重新采集数据或人工审核。'
}

💡 个性化建议:
   1. {
    '可以放心使用该分类结果进行后续分析。' if confidence > 0.8
    else '建议结合其他指标进行综合判断。' if confidence > 0.6
    else '建议人工复核或重新测量。' if confidence > 0.4
    else '强烈建议重新采集数据。'
}
   2. 若用于科研分析，请标注置信度水平以便后续统计分析。
   3. 建议定期更新模型以适应新的数据特征。

---
报告由DeepLearning AI分析引擎自动生成
"""
        
        self.ai_analysis_text.config(state=tk.NORMAL)
        self.ai_analysis_text.delete(1.0, tk.END)
        self.ai_analysis_text.insert(tk.END, analysis_text)
        self.ai_analysis_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = WaveformClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()