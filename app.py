import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from waveform_classifier import train_model, predict_waveform

class WaveformClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("角度波形分类系统")
        self.root.geometry("800x600")
        
        # 创建 Notebook 控件用于选项卡
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建训练选项卡
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="模型训练")
        self.setup_train_tab()
        
        # 创建预测选项卡
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="波形预测")
        self.setup_predict_tab()
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_train_tab(self):
        # 数据目录选择
        data_dir_frame = ttk.Frame(self.train_frame)
        data_dir_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(data_dir_frame, text="数据目录:").pack(side=tk.LEFT)
        self.data_dir_var = tk.StringVar()
        self.data_dir_entry = ttk.Entry(data_dir_frame, textvariable=self.data_dir_var, width=50)
        self.data_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(data_dir_frame, text="浏览", command=self.browse_data_dir).pack(side=tk.LEFT, padx=5)
        
        # 模型保存路径
        model_path_frame = ttk.Frame(self.train_frame)
        model_path_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_path_frame, text="模型保存路径:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value="waveform_model.pth")
        self.model_path_entry = ttk.Entry(model_path_frame, textvariable=self.model_path_var, width=50)
        self.model_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="浏览", command=self.browse_model_path).pack(side=tk.LEFT, padx=5)
        
        # 训练参数
        params_frame = ttk.LabelFrame(self.train_frame, text="训练参数")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Epochs
        epoch_frame = ttk.Frame(params_frame)
        epoch_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(epoch_frame, text="训练轮数:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(epoch_frame, textvariable=self.epochs_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # 训练按钮
        button_frame = ttk.Frame(self.train_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        self.train_button = ttk.Button(button_frame, text="开始训练", command=self.start_training)
        self.train_button.pack(side=tk.LEFT)
        
        # 训练日志
        log_frame = ttk.LabelFrame(self.train_frame, text="训练日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.train_log_text = tk.Text(log_frame, height=15)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.train_log_text.yview)
        self.train_log_text.configure(yscrollcommand=scrollbar.set)
        self.train_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_predict_tab(self):
        # 模型路径
        model_frame = ttk.Frame(self.predict_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="模型路径:").pack(side=tk.LEFT)
        self.pred_model_path_var = tk.StringVar(value="waveform_model.pth")
        self.pred_model_path_entry = ttk.Entry(model_frame, textvariable=self.pred_model_path_var, width=50)
        self.pred_model_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览", command=self.browse_pred_model_path).pack(side=tk.LEFT, padx=5)
        
        # CSV文件路径
        csv_frame = ttk.Frame(self.predict_frame)
        csv_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(csv_frame, text="CSV文件:").pack(side=tk.LEFT)
        self.csv_path_var = tk.StringVar()
        self.csv_path_entry = ttk.Entry(csv_frame, textvariable=self.csv_path_var, width=50)
        self.csv_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(csv_frame, text="浏览", command=self.browse_csv_file).pack(side=tk.LEFT, padx=5)
        
        # 预测按钮
        predict_btn_frame = ttk.Frame(self.predict_frame)
        predict_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.predict_button = ttk.Button(predict_btn_frame, text="开始预测", command=self.start_prediction)
        self.predict_button.pack(side=tk.LEFT)
        
        # 预测结果
        result_frame = ttk.LabelFrame(self.predict_frame, text="预测结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 主要预测结果
        main_result_frame = ttk.Frame(result_frame)
        main_result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(main_result_frame, text="预测类别:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.pred_class_var = tk.StringVar()
        ttk.Label(main_result_frame, textvariable=self.pred_class_var, font=("Arial", 12, "bold")).grid(
            row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(main_result_frame, text="置信度:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.StringVar()
        ttk.Label(main_result_frame, textvariable=self.confidence_var, font=("Arial", 12, "bold")).grid(
            row=1, column=1, sticky=tk.W, padx=5)
        
        # 详细概率结果
        detail_frame = ttk.LabelFrame(result_frame, text="各类别概率")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_tree = ttk.Treeview(detail_frame, columns=("probability"), show="tree headings")
        self.result_tree.heading("#0", text="类别")
        self.result_tree.heading("probability", text="概率")
        self.result_tree.column("#0", width=200)
        self.result_tree.column("probability", width=100)
        
        tree_scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scrollbar.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
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
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.csv_path_var.set(file_path)
    
    def start_training(self):
        if not self.data_dir_var.get():
            messagebox.showerror("错误", "请选择数据目录")
            return
        
        self.train_button.config(state=tk.DISABLED)
        self.status_var.set("正在训练模型...")
        
        # 在后台线程中运行训练
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
    
    def run_training(self):
        try:
            train_model(
                data_dir=self.data_dir_var.get(),
                model_save_path=self.model_path_var.get(),
                epochs=int(self.epochs_var.get())
            )
            self.root.after(0, lambda: self.train_log_text.insert(tk.END, "训练完成!\n"))
            self.root.after(0, lambda: messagebox.showinfo("成功", "模型训练完成!"))
        except Exception as e:
            self.root.after(0, lambda: self.train_log_text.insert(tk.END, f"训练出错: {str(e)}\n"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"训练过程中出现错误:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("就绪"))
    
    def start_prediction(self):
        if not self.pred_model_path_var.get():
            messagebox.showerror("错误", "请选择模型文件")
            return
        
        if not self.csv_path_var.get():
            messagebox.showerror("错误", "请选择CSV文件")
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
            for label, prob in result["all_probabilities"].items():
                self.root.after(0, lambda l=label, p=prob: self.result_tree.insert(
                    "", tk.END, text=l, values=(f"{p:.4f}",)))
            
            self.root.after(0, lambda: messagebox.showinfo("成功", "预测完成!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"预测过程中出现错误:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.predict_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_var.set("就绪"))

def main():
    root = tk.Tk()
    app = WaveformClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()