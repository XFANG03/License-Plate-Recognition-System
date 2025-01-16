import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from main import PlateDetectionSystem


class ModernButton(ttk.Button):
    """自定义现代风格按钮"""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, e):
        self['style'] = 'Accent.TButton'

    def _on_leave(self, e):
        self['style'] = 'TButton'


class PlateRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别系统 - 智能识别")
        self.root.geometry("1000x800")

        # 设置主题和样式
        self.setup_styles()

        # 初始化车牌检测系统
        self.detection_system = PlateDetectionSystem()

        # 创建主容器
        self.create_main_container()

        # 创建header部分
        self.create_header()

        # 创建主要内容区域
        self.create_content_area()

        # 创建状态栏
        self.create_status_bar()

        # 绑定键盘快捷键
        self.bind_shortcuts()

        # 初始化图片预览
        self.preview_image = None

    def setup_styles(self):
        """设置自定义样式"""
        style = ttk.Style()

        # 配置全局字体和颜色
        style.configure('.',
                        font=('Microsoft YaHei UI', 10),
                        background='#f0f0f0')

        # 主标题样式
        style.configure('Header.TLabel',
                        font=('Microsoft YaHei UI', 24, 'bold'),
                        foreground='#2c3e50',
                        padding=20)

        # 按钮样式
        style.configure('TButton',
                        padding=10,
                        font=('Microsoft YaHei UI', 10))

        # 强调按钮样式
        style.configure('Accent.TButton',
                        background='#3498db',
                        foreground='white')

        # 结果框样式
        style.configure('Result.TLabelframe',
                        background='white',
                        padding=15)

        style.configure('Result.TLabelframe.Label',
                        font=('Microsoft YaHei UI', 11, 'bold'),
                        foreground='#2c3e50')

        # 状态栏样式
        style.configure('Status.TLabel',
                        background='#2c3e50',
                        foreground='white',
                        padding=5)

    def create_main_container(self):
        """创建主容器"""
        self.main_container = ttk.Frame(self.root, padding="20")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def create_header(self):
        """创建头部区域"""
        # 标题
        header = ttk.Label(
            self.main_container,
            text="智能车牌识别系统",
            style='Header.TLabel'
        )
        header.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 操作按钮区
        btn_frame = ttk.Frame(self.main_container)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        self.select_button = ModernButton(
            btn_frame,
            text="选择图片",
            command=self.select_file,
            width=20
        )
        self.select_button.grid(row=0, column=0, padx=5)

    def create_content_area(self):
        """创建主要内容区域"""
        # 创建预览区域
        preview_frame = ttk.LabelFrame(
            self.main_container,
            text="图片预览",
            style='Result.TLabelframe'
        )
        preview_frame.grid(row=2, column=0, sticky='nsew', padx=(0, 10))

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, padx=10, pady=10)

        # 创建结果显示区域
        result_frame = ttk.LabelFrame(
            self.main_container,
            text="识别结果",
            style='Result.TLabelframe'
        )
        result_frame.grid(row=2, column=1, sticky='nsew')

        # 创建一个滚动画布来容纳多个车牌结果
        canvas = tk.Canvas(result_frame, background='#f8f9fa')
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=canvas.yview)
        self.results_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)

        # 布局滚动组件
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # 创建窗口来显示结果框架
        canvas.create_window((0, 0), window=self.results_frame, anchor="nw")

        # 配置滚动区域
        self.results_frame.bind("<Configure>",
                                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # 设置列权重
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_label = ttk.Label(
            self.main_container,
            text="就绪",
            style='Status.TLabel'
        )
        self.status_label.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(20, 0))

    def bind_shortcuts(self):
        """绑定键盘快捷键"""
        self.root.bind('<Control-o>', lambda e: self.select_file())
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def select_file(self):
        """处理文件选择"""
        file_path = filedialog.askopenfilename(
            title="选择车牌图片",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.update_status("正在处理图片...")
            self.show_preview(file_path)
            self.process_image(file_path)

    def show_preview(self, image_path: str):
        """显示原始图片预览"""
        # 读取并调整原始图片大小
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 计算调整后的尺寸
        max_size = (400, 400)
        img_pil.thumbnail(max_size, Image.LANCZOS)

        # 转换为PhotoImage并显示
        self.preview_image = ImageTk.PhotoImage(img_pil)
        self.preview_label.configure(image=self.preview_image)

    def process_image(self, image_path: str):
        """处理选中的图片"""
        try:
            # 使用车牌检测系统处理图片
            results = self.detection_system.detect(image_path)

            if results:
                # 清除之前的结果
                for widget in self.results_frame.winfo_children():
                    widget.destroy()

                # 显示结果
                for i, (plate_img, plate_number) in enumerate(results, 1):
                    # 为每个车牌创建一个框架
                    plate_frame = ttk.Frame(self.results_frame)
                    plate_frame.pack(pady=10, padx=5, fill='x')

                    # 转换车牌图片格式
                    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
                    plate_img_pil = Image.fromarray(plate_img_rgb)

                    # 调整车牌图片大小
                    display_size = (300, 100)
                    plate_img_pil = plate_img_pil.resize(display_size, Image.LANCZOS)

                    # 创建并保存PhotoImage（需要保持引用）
                    plate_img_tk = ImageTk.PhotoImage(plate_img_pil)
                    plate_label = ttk.Label(plate_frame, image=plate_img_tk)
                    plate_label.image = plate_img_tk  # 保持引用
                    plate_label.pack(pady=5)

                    # 添加识别结果文本
                    plate_type = "港澳车牌" if len(plate_number) < 7 else "大陆车牌"
                    result_text = f"车牌 {i}\n类型: {plate_type}\n识别结果: {plate_number}"
                    text_label = ttk.Label(
                        plate_frame,
                        text=result_text,
                        font=('Microsoft YaHei UI', 12),
                        background='#f8f9fa'
                    )
                    text_label.pack(pady=5)

                    # 添加分隔线（除了最后一个结果）
                    if i < len(results):
                        separator = ttk.Separator(self.results_frame, orient='horizontal')
                        separator.pack(fill='x', padx=20, pady=10)

                self.update_status("处理完成")
            else:
                # 清除之前的结果
                for widget in self.results_frame.winfo_children():
                    widget.destroy()

                # 显示错误信息
                error_label = ttk.Label(
                    self.results_frame,
                    text="未能识别车牌\n请确保图片中包含清晰的车牌。",
                    font=('Microsoft YaHei UI', 12),
                    background='#f8f9fa'
                )
                error_label.pack(pady=20)
                self.update_status("未检测到车牌")

        except Exception as e:
            # 清除之前的结果
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # 显示错误信息
            error_label = ttk.Label(
                self.results_frame,
                text=f"发生错误\n{str(e)}",
                font=('Microsoft YaHei UI', 12),
                background='#f8f9fa'
            )
            error_label.pack(pady=20)
            self.update_status("处理出错")

    def update_status(self, message: str):
        """更新状态栏消息"""
        self.status_label.config(text=message)
        self.root.update()


def main():
    root = tk.Tk()
    root.configure(bg='#f0f0f0')  # 设置背景色
    app = PlateRecognitionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()