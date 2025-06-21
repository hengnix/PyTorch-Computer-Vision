import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict


class FruitVegClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("水果和蔬菜分类器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        # 默认模型路径
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "model.pth"
        )
        self.is_predicting = False
        self.last_prediction = None
        self.current_image = None
        self.photo = None

        # 创建界面
        self.create_widgets()

        # 检查模型是否存在
        self.check_model()

    def create_widgets(self):
        """创建 GUI 界面组件"""
        # 顶部菜单
        self.create_menu()

        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧面板（图像和上传）
        left_frame = ttk.LabelFrame(main_frame, text="图像", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 图像显示区域
        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # 创建图像标签，设置为可调整大小
        self.image_label = ttk.Label(
            self.image_frame, text="请上传图像", anchor="center", padding="20"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 绑定窗口调整事件，确保图像能适应窗口大小
        self.image_frame.bind("<Configure>", self.on_frame_resize)

        # 上传按钮
        self.upload_button = ttk.Button(
            left_frame, text="上传图像", command=self.upload_image
        )
        self.upload_button.pack(pady=10)

        # 右侧面板（结果和评估）
        right_frame = ttk.LabelFrame(main_frame, text="结果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 预测结果表格
        self.result_frame = ttk.Frame(right_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("排名", "类别", "置信度")
        self.result_tree = ttk.Treeview(
            self.result_frame, columns=columns, show="headings"
        )

        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=150, anchor="center")

        self.result_tree.pack(fill=tk.BOTH, expand=True)

        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        self.status_label = ttk.Label(
            self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        """创建顶部菜单"""
        menubar = tk.Menu(self.root)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="上传图像", command=self.upload_image, accelerator="Ctrl+U"
        )
        file_menu.add_command(
            label="加载模型", command=self.load_model, accelerator="Ctrl+M"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="退出", command=self.root.quit, accelerator="Ctrl+Q"
        )
        menubar.add_cascade(label="文件", menu=file_menu)

        # 查看菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(
            label="查看评估结果", command=self.show_evaluation_results
        )
        menubar.add_cascade(label="查看", menu=view_menu)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 设置快捷键
        self.root.bind("<Control-u>", lambda event: self.upload_image())
        self.root.bind("<Control-m>", lambda event: self.load_model())
        self.root.bind("<Control-q>", lambda event: self.root.quit())

        self.root.config(menu=menubar)

    def check_model(self):
        """检查模型文件是否存在"""
        if os.path.exists(self.model_path):
            self.status_var.set(f"已加载模型: {self.model_path}")
        else:
            self.status_var.set(f"警告: 模型文件 {self.model_path} 不存在")
            messagebox.showwarning(
                "模型缺失",
                f"模型文件 {self.model_path} 不存在\n请先训练模型或加载现有模型",
            )

    def upload_image(self):
        """上传图像并进行预测"""
        if self.is_predicting:
            return

        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")],
        )

        if file_path:
            self.status_var.set("正在预测...")
            self.is_predicting = True
            self.upload_button.config(state=tk.DISABLED)

            # 清空结果表格
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)

            # 在新线程中执行预测
            thread = threading.Thread(target=self.predict_image, args=(file_path,))
            thread.daemon = True
            thread.start()

    def predict_image(self, file_path):
        """在后台线程中执行预测"""
        try:
            # 显示上传的图像
            image = Image.open(file_path)
            self.current_image = image  # 保存图像引用

            # 更新图像显示
            self.root.after(0, self.update_image_display)

            # 执行预测
            predictions = predict(file_path, self.model_path, top_k=5)
            self.last_prediction = predictions

            # 更新结果表格
            self.root.after(0, self.update_result_table, predictions)

            self.status_var.set(f"预测完成: {file_path}")

        except Exception as e:
            self.root.after(
                0, messagebox.showerror, "预测错误", f"预测过程中出错: {str(e)}"
            )
            self.status_var.set("预测出错")
        finally:
            self.root.after(0, lambda: self.upload_button.config(state=tk.NORMAL))
            self.is_predicting = False

    def update_image_display(self):
        """更新图像显示，确保适应标签大小"""
        if not self.current_image:
            return

        # 获取标签的当前尺寸
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        # 如果标签尺寸太小，使用默认值
        if label_width < 10 or label_height < 10:
            label_width, label_height = 400, 400

        # 调整图像大小
        resized_image = self.current_image.copy()
        resized_image.thumbnail((label_width, label_height), Image.LANCZOS)

        # 更新图像显示
        self.photo = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # 保持引用，防止被垃圾回收

    def on_frame_resize(self, event):
        """处理框架大小变化事件，调整图像大小"""
        if self.current_image and hasattr(self, "photo"):
            self.update_image_display()

    def update_result_table(self, predictions):
        """更新预测结果表格"""
        # 清空现有结果
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        # 添加新结果
        for i, pred in enumerate(predictions, 1):
            self.result_tree.insert(
                "", tk.END, values=(i, pred["class"], pred["confidence"])
            )

    def load_model(self):
        """加载模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件", filetypes=[("模型文件", "*.pth")]
        )

        if file_path:
            self.model_path = file_path
            self.status_var.set(f"已加载模型: {self.model_path}")
            messagebox.showinfo("模型加载成功", f"已加载模型: {self.model_path}")

    def show_evaluation_results(self):
        """显示模型评估结果"""
        try:
            # 查找最新的评估结果文件
            results_files = [
                f
                for f in os.listdir(".")
                if f.startswith("evaluation_results_") and f.endswith(".json")
            ]
            if not results_files:
                raise FileNotFoundError("未找到评估结果文件")

            # 按时间排序，获取最新的文件
            results_files.sort(reverse=True)
            latest_results_file = results_files[0]  # 尝试导入 ResultsViewer 类
            try:
                from src.gui_results import ResultsViewer
            except ImportError:
                # 如果无法导入，尝试其他方式
                try:
                    import gui_results

                    ResultsViewer = gui_results.ResultsViewer
                except ImportError:
                    raise ImportError("无法找到 ResultsViewer 类")

            # 打开结果查看窗口
            results_window = tk.Toplevel(self.root)
            ResultsViewer(results_window, latest_results_file)

        except Exception as e:
            messagebox.showerror("错误", f"无法显示评估结果: {str(e)}")

    def show_about(self):
        """显示关于对话框"""
        about_window = tk.Toplevel(self.root)
        about_window.title("关于")
        about_window.geometry("400x300")
        about_window.resizable(False, False)

        ttk.Label(
            about_window, text="水果和蔬菜分类器", font=("Arial", 16, "bold")
        ).pack(pady=20)
        ttk.Label(about_window, text="版本：1.0.0").pack(pady=5)
        ttk.Label(about_window, text="基于 PyTorch 和深度学习技术").pack(pady=5)
        ttk.Label(about_window, text="用于识别常见水果和蔬菜").pack(pady=5)
        ttk.Label(about_window, text="© 2025 水果和蔬菜分类器团队").pack(pady=20)

        ttk.Button(about_window, text="确定", command=about_window.destroy).pack(
            pady=10
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = FruitVegClassifierGUI(root)
    root.mainloop()
