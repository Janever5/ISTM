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

class WaveformClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("膝关节康复角度波形分类系统")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # 设置样式
        self.setup_styles()
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        self.notebook.add(self.predict_frame, text="🔮 波形预测")
        self.setup_predict_tab()
        
        # 创建可视化选项卡
        self.visualize_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualize_frame, text="📊 数据可视化")
        self.setup_visualize_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 膝关节康复角度波形分类系统")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Header.TLabel', font=('微软雅黑', 16, 'bold'), foreground='#2c3e50', background='#f0f0f0')
        style.configure('Title.TLabel', font=('微软雅黑', 12, 'bold'), foreground='#3498db')
        style.configure('Custom.TNotebook', background='#f0f0f0')
        style.configure('Custom.TNotebook.Tab', font=('微软雅黑', 10), padding=[10, 5])
        style.map('Custom.TNotebook.Tab', background=[('selected', '#3498db')], foreground=[('selected', 'white')])
        
        # 按钮样式
        style.configure('Action.TButton', font=('微软雅黑', 10, 'bold'), padding=6)
        style.configure('Primary.TButton', background='#3498db', foreground='white')
        style.map('Primary.TButton', background=[('active', '#2980b9')])
        
        style.configure('Secondary.TButton', background='#95a5a6', foreground='white')
        style.map('Secondary.TButton', background=[('active', '#7f8c8d')])
        
        style.configure('Success.TButton', background='#2ecc71', foreground='white')
        style.map('Success.TButton', background=[('active', '#27ae60')])
        
        style.configure('Danger.TButton', background='#e74c3c', foreground='white')
        style.map('Danger.TButton', background=[('active', '#c0392b')])
    
    def create_header(self):
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🏥 膝关节康复角度波形分类系统", style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        version_label = ttk.Label(header_frame, text="v1.0", font=('微软雅黑', 9), foreground='#7f8c8d')
        version_label.pack(side=tk.RIGHT)
    
    def setup_train_tab(self):
        # 主训练框架
        train_main_frame = ttk.Frame(self.train_frame)
        train_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 配置区域
        config_frame = ttk.LabelFrame(train_main_frame, text="⚙️ 训练配置", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 数据目录选择
        data_dir_frame = ttk.LabelFrame(config_frame, text="📂 数据源", padding=5)
        data_dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(data_dir_frame, text="数据目录:").pack(anchor=tk.W)
        self.data_dir_var = tk.StringVar()
        self.data_dir_entry = ttk.Entry(data_dir_frame, textvariable=self.data_dir_var, width=30)
        self.data_dir_entry.pack(fill=tk.X, pady=5)
        ttk.Button(data_dir_frame, text="📁 浏览", command=self.browse_data_dir, style='Secondary.TButton').pack(fill=tk.X)
        
        # 模型保存路径
        model_path_frame = ttk.LabelFrame(config_frame, text="💾 模型保存", padding=5)
        model_path_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_path_frame, text="模型路径:").pack(anchor=tk.W)
        self.model_path_var = tk.StringVar(value="waveform_model.pth")
        self.model_path_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var, width=30)
        self.model_path_entry.pack(fill=tk.X, pady=5)
        ttk.Button(model_path_frame, text="📁 浏览", command=self.browse_model_path, style='Secondary.TButton').pack(fill=tk.X)
        
        # 训练参数
        params_frame = ttk.LabelFrame(config_frame, text="🎛️ 训练参数", padding=5)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Epochs
        ttk.Label(params_frame, text="训练轮数:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=20).pack(fill=tk.X, pady=5)
        
        # Batch Size
        ttk.Label(params_frame, text="批处理大小:").pack(anchor=tk.W, pady=(10, 0))
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=20).pack(fill=tk.X, pady=5)
        
        # 学习率
        ttk.Label(params_frame, text="学习率:").pack(anchor=tk.W, pady=(10, 0))
        self.learning_rate_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=20).pack(fill=tk.X, pady=5)
        
        # 训练按钮
        self.train_button = ttk.Button(config_frame, text="🚀 开始训练", command=self.start_training, style='Primary.TButton')
        self.train_button.pack(fill=tk.X, pady=(20, 0))
        
        # 右侧面板 - 日志和进度
        right_frame = ttk.Frame(train_main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 训练状态
        status_frame = ttk.LabelFrame(right_frame, text="📈 训练状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_text_var = tk.StringVar()
        self.status_text_var.set("等待开始训练...")
        status_label = ttk.Label(status_frame, textvariable=self.status_text_var)
        status_label.pack()
        
        # 训练日志
        log_frame = ttk.LabelFrame(right_frame, text="📋 训练日志", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.train_log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.train_log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_predict_tab(self):
        # 主预测框架
        predict_main_frame = ttk.Frame(self.predict_frame)
        predict_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 预测配置
        config_frame = ttk.LabelFrame(predict_main_frame, text="⚙️ 预测配置", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 模型路径
        model_frame = ttk.LabelFrame(config_frame, text="🤖 模型文件", padding=5)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(model_frame, text="模型路径:").pack(anchor=tk.W)
        self.pred_model_path_var = tk.StringVar(value="waveform_model.pth")
        self.pred_model_path_entry = ttk.Entry(model_frame, textvariable=self.pred_model_path_var, width=30)
        self.pred_model_path_entry.pack(fill=tk.X, pady=5)
        ttk.Button(model_frame, text="📁 浏览", command=self.browse_pred_model_path, style='Secondary.TButton').pack(fill=tk.X)
        
        # 文件路径
        file_frame = ttk.LabelFrame(config_frame, text="📄 待预测文件", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="数据文件:").pack(anchor=tk.W)
        self.csv_path_var = tk.StringVar()
        self.csv_path_entry = ttk.Entry(file_frame, textvariable=self.csv_path_var, width=30)
        self.csv_path_entry.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="📁 浏览", command=self.browse_csv_file, style='Secondary.TButton').pack(fill=tk.X)
        
        # 预测按钮
        self.predict_button = ttk.Button(config_frame, text="🔍 开始预测", command=self.start_prediction, style='Success.TButton')
        self.predict_button.pack(fill=tk.X, pady=(20, 0))
        
        # 右侧面板 - 预测结果
        result_frame = ttk.Frame(predict_main_frame)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 主要预测结果
        main_result_frame = ttk.LabelFrame(result_frame, text="🎯 预测结果", padding=10)
        main_result_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 预测类别
        pred_class_frame = ttk.Frame(main_result_frame)
        pred_class_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pred_class_frame, text="预测类别:", font=('微软雅黑', 10, 'bold')).pack(side=tk.LEFT)
        self.pred_class_var = tk.StringVar()
        pred_class_label = ttk.Label(pred_class_frame, textvariable=self.pred_class_var, font=('微软雅黑', 12, 'bold'), foreground='#3498db')
        pred_class_label.pack(side=tk.RIGHT)
        
        # 置信度
        confidence_frame = ttk.Frame(main_result_frame)
        confidence_frame.pack(fill=tk.X, pady=5)
        ttk.Label(confidence_frame, text="置信度:", font=('微软雅黑', 10, 'bold')).pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar()
        confidence_label = ttk.Label(confidence_frame, textvariable=self.confidence_var, font=('微软雅黑', 12, 'bold'), foreground='#2ecc71')
        confidence_label.pack(side=tk.RIGHT)
        
        # 详细概率结果
        detail_frame = ttk.LabelFrame(result_frame, text="📊 各类别概率分布", padding=5)
        detail_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview显示详细结果
        columns = ('类别', '概率')
        self.result_tree = ttk.Treeview(detail_frame, columns=columns, show='headings', height=10)
        self.result_tree.heading('类别', text='类别')
        self.result_tree.heading('概率', text='概率')
        self.result_tree.column('类别', width=150)
        self.result_tree.column('概率', width=100)
        
        tree_scrollbar_y = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        tree_scrollbar_x = ttk.Scrollbar(detail_frame, orient=tk.HORIZONTAL, command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_visualize_tab(self):
        # 可视化框架
        visualize_main_frame = ttk.Frame(self.visualize_frame)
        visualize_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(visualize_main_frame, text="🎛️ 控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件选择
        file_select_frame = ttk.Frame(control_frame)
        file_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_select_frame, text="数据文件:").pack(side=tk.LEFT)
        self.vis_file_path_var = tk.StringVar()
        self.vis_file_entry = ttk.Entry(file_select_frame, textvariable=self.vis_file_path_var, width=50)
        self.vis_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_select_frame, text="📁 浏览", command=self.browse_vis_file, style='Secondary.TButton').pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(file_select_frame, text="👁️ 查看", command=self.visualize_waveform, style='Primary.TButton').pack(side=tk.LEFT, padx=(5, 0))
        
        # 图表区域
        chart_frame = ttk.LabelFrame(visualize_main_frame, text="📉 波形图表", padding=5)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始图表
        self.ax.set_title("波形数据可视化")
        self.ax.set_xlabel("时间点")
        self.ax.set_ylabel("电流值")
        self.canvas.draw()
    
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
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
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
            self.ax.plot(current, linewidth=1.5)
            self.ax.set_title(f"波形数据可视化 - {Path(file_path).name}")
            self.ax.set_xlabel("时间点")
            self.ax.set_ylabel("电流/电阻值")
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
        
        # 在后台线程中运行训练
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def run_training(self):
        try:
            self.root.after(0, lambda: self.append_log("开始训练模型...\n"))
            
            train_model(
                data_dir=self.data_dir_var.get(),
                model_save_path=self.model_path_var.get(),
                epochs=int(self.epochs_var.get()),
                batch_size=int(self.batch_size_var.get())
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
            
            # 清空并更新详细结果
            self.root.after(0, self.result_tree.delete(*self.result_tree.get_children()))
            
            # 按概率排序
            sorted_results = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
            
            for label, prob in sorted_results:
                prob_str = f"{prob:.4f}"
                self.root.after(0, lambda l=label, p=prob_str: self.result_tree.insert("", tk.END, values=(l, p)))
            
            self.root.after(0, lambda: messagebox.showinfo("成功", "预测完成!"))
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("错误", f"预测过程中出现错误:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("就绪"))

def main():
    root = tk.Tk()
    app = WaveformClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()