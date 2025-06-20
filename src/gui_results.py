import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ResultsViewer:
    def __init__(self, root, results_file=None):
        self.root = root
        self.root.title("模型评估结果")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # 如果未指定结果文件，尝试查找最新的
        if results_file:
            self.results_file = results_file
        else:
            self.results_file = self.find_latest_results_file()

        # 加载评估结果
        self.load_evaluation_results()

        # 创建界面
        self.create_widgets()

    def find_latest_results_file(self):
        """查找最新的评估结果文件"""
        results_files = [
            f
            for f in os.listdir(".")
            if f.startswith("evaluation_results_") and f.endswith(".json")
        ]
        if not results_files:
            messagebox.showerror("错误", "未找到评估结果文件\n请先训练模型生成评估结果")
            self.root.destroy()
            return None

        # 按时间排序，获取最新的文件
        results_files.sort(reverse=True)
        return results_files[0]

    def load_evaluation_results(self):
        """加载评估结果文件"""
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            self.classes = self.results["classes"]
        except Exception as e:
            messagebox.showerror("错误", f"加载评估结果失败: {str(e)}")
            self.root.destroy()

    def create_widgets(self):
        """创建界面组件"""
        # 顶部标题
        title_label = ttk.Label(
            self.root,
            text=f"模型评估结果 - {self.results_file}",
            font=("微软雅黑", 14, "bold"),
        )
        title_label.pack(pady=10)

        # 选项卡控件
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 测试集结果选项卡
        self.test_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.test_frame, text="测试集结果")
        self.create_result_tab(self.test_frame, self.results["test"])

        # 验证集结果选项卡
        self.val_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.val_frame, text="验证集结果")
        self.create_result_tab(self.val_frame, self.results["val"])

        # 训练指标选项卡
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="训练指标")
        self.create_metrics_tab()

        # 总体统计信息选项卡
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="总体统计")
        self.create_summary_tab()

    def create_result_tab(self, parent, results):
        """创建结果选项卡（含表格和图表）"""
        # 创建顶部框架（图表）
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # 创建底部框架（表格）
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建 F1 分数柱状图
        fig1 = plt.Figure(figsize=(6, 4), dpi=100)
        ax1 = fig1.add_subplot(111)

        classes = [c for c in results if c != "overall"]
        f1_scores = [results[c]["f1-score"] for c in classes]

        ax1.bar(classes, f1_scores, color="skyblue")
        ax1.set_title("各类别 F1 分数")
        ax1.set_xlabel("类别")
        ax1.set_ylabel("F1 分数")
        ax1.tick_params(axis="x", rotation=90)

        canvas1 = FigureCanvasTkAgg(fig1, master=top_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建样本分布柱状图
        fig2 = plt.Figure(figsize=(6, 4), dpi=100)
        ax2 = fig2.add_subplot(111)

        support = [results[c]["support"] for c in classes]

        ax2.bar(classes, support, color="lightgreen")
        ax2.set_title("各类别样本数")
        ax2.set_xlabel("类别")
        ax2.set_ylabel("样本数")
        ax2.tick_params(axis="x", rotation=90)

        canvas2 = FigureCanvasTkAgg(fig2, master=top_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建表格
        columns = ("类别", "精确率", "召回率", "F1 分数", "样本数")
        self.tree = ttk.Treeview(bottom_frame, columns=columns, show="headings")

        # 设置列宽和标题
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)

        # 添加数据
        for class_name in classes:
            data = results[class_name]
            self.tree.insert(
                "",
                tk.END,
                values=(
                    class_name,
                    f"{data['precision']:.4f}",
                    f"{data['recall']:.4f}",
                    f"{data['f1-score']:.4f}",
                    data["support"],
                ),
            )

        # 添加总体数据
        overall = results["overall"]
        self.tree.insert(
            "",
            tk.END,
            values=(
                "总体",
                f"{overall['macro_avg']['precision']:.4f}",
                f"{overall['macro_avg']['recall']:.4f}",
                f"{overall['accuracy']:.4f}",
                "N/A",
            ),
            iid="overall",
        )

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.tag_configure("overall", background="lightgray")

    def create_metrics_tab(self):
        """创建训练指标选项卡"""
        frame = ttk.Frame(self.metrics_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        metrics = self.results["training_metrics"]

        # 创建损失曲线
        fig1 = plt.Figure(figsize=(8, 6), dpi=100)
        ax1 = fig1.add_subplot(111)

        ax1.plot(metrics["train_losses"], label="训练损失")
        ax1.plot(metrics["val_losses"], label="验证损失")
        ax1.set_title("训练和验证损失")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("损失")
        ax1.legend()
        ax1.grid(True)

        canvas1 = FigureCanvasTkAgg(fig1, master=frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
