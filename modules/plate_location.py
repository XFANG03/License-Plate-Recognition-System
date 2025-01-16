import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImagePreprocessor:
    def __init__(self):
        pass

    def gray_guss(self, image):
        """灰度化和高斯滤波"""
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def process(self, image):
        """
        图像预处理主函数，包含可视化
        Args:
            image: 输入图像数组（已经读取的图像）
        Returns:
            处理后的图像
        """
        # 复制图像
        image = image.copy()

        # 灰度化和高斯滤波
        gray_image = self.gray_guss(image)

        # Sobel边缘检测
        Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
        absX = cv2.convertScaleAbs(Sobel_x)
        image = absX

        # 图像阈值化操作
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

        return image


class PlateLocator:
    def __init__(self, debug=True):
        self.preprocessor = ImagePreprocessor()
        self.debug = debug
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 存储所有处理步骤的图像
        self.process_images = []
        self.process_titles = []

    def add_process_image(self, title, image, is_bgr=False):
        """添加处理步骤图像到列表"""
        if is_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.process_images.append(image)
        self.process_titles.append(title)

    def show_all_process(self):
        """显示所有处理步骤，以4行2列的方式排列"""
        n = len(self.process_images)
        if n == 0:
            return

        # 固定为4行2列布局
        rows, cols = 4, 2
        plt.figure(figsize=(12, 20))  # 调整图像大小以适应新布局

        for i, (img, title) in enumerate(zip(self.process_images, self.process_titles)):
            if i >= rows * cols:  # 如果图像数量超过8张，跳过多余的
                print(f"警告：超过{rows * cols}张图像，只显示前{rows * cols}张")
                break

            plt.subplot(rows, cols, i + 1)
            plt.title(title)
            if len(img.shape) == 3:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap='gray')
            plt.axis('off')

        # 如果图像数量少于8张，用空白填充剩余位置
        for i in range(len(self.process_images), rows * cols):
            plt.subplot(rows, cols, i + 1)
            plt.axis('off')

        plt.tight_layout(pad=3.0)  # 增加间距
        plt.show()

        # 清空图像列表，为下一次处理做准备
        self.process_images = []
        self.process_titles = []
    def locate(self, preprocessed_image, origin_image):
        """
        定位车牌位置
        """
        # 创建副本
        image = preprocessed_image.copy()

        if self.debug:
            print("1. 开始形态学处理...")

        # 添加原始预处理图像
        self.add_process_image("预处理结果", image)

        small_kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        image = cv2.erode(image, small_kernel_x)
        image = cv2.dilate(image, small_kernel_x)
        if self.debug:
            print("2. 完成小的x方向开操作")
            self.add_process_image("x方向开操作", image)

        final_kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        image = cv2.dilate(image, final_kernel_y)
        if self.debug:
            print("3. 最终Y方向调整完成")
            self.add_process_image("Y方向调整", image)

        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=1)
        if self.debug:
            print("4. 第一次形态学闭操作完成")
            self.add_process_image("形态学闭操作", image)

        kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

        image = cv2.dilate(image, kernelX)
        image = cv2.erode(image, kernelX)
        if self.debug:
            print("5. X方向闭操作完成")
            self.add_process_image("X方向闭操作", image)

        image = cv2.erode(image, kernelY)
        image = cv2.dilate(image, kernelY)
        if self.debug:
            print("6. Y方向开操作完成")
            self.add_process_image("Y方向开操作", image)

        image = cv2.medianBlur(image, 9)
        if self.debug:
            print("7. 中值滤波完成")
            self.add_process_image("中值滤波", image)

        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            print(f"8. 找到 {len(contours)} 个轮廓")
            debug_img = origin_image.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            self.add_process_image("检测到的轮廓", debug_img, True)
            # 显示所有处理步骤
            self.show_all_process()

        # 遍历所有轮廓
        for i, item in enumerate(contours):
            area = cv2.contourArea(item)
            min_area = 2000
            max_area = 30000

            if self.debug:
                print(f"轮廓 {i + 1} 面积: {area}")

            if min_area <= area <= max_area:
                rect = cv2.boundingRect(item)
                x, y, width, height = rect

                ratio = width / height
                if (width > (height * 2)) and (width < (height * 4.5)):
                    if self.debug:
                        print(f"找到符合条件的轮廓 {i + 1}")
                        print(f"   位置: x={x}, y={y}, width={width}, height={height}")
                        print(f"   长宽比: {ratio:.2f}")
                        print(f"   面积: {area}")

                    plate_image = origin_image[y:y + height, x:x + width]
                    if self.debug:
                        # 单独显示最终车牌
                        plt.figure(figsize=(8, 4))
                        plt.title("最终定位到的车牌")
                        plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
                        plt.axis('off')
                        plt.show()
                    return plate_image

        if self.debug:
            print("未找到符合条件的车牌区域")
        return None
