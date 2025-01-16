import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class PlateRefiner:
    """车牌精确边界提取"""

    def __init__(self):
        self.debug = True
        self.min_change_count = 12
        self.max_change_count = 60

    def _remove_rivets(self, plate: np.ndarray) -> Optional[np.ndarray]:
        """
        通过分析颜色跳变次数清除车牌上的铆钉
        Args:
            plate: 输入的二值化车牌图像
        Returns:
            处理后的图像，处理失败返回None
        """

        plate = plate.copy()
        rows, cols = plate.shape

        if self.debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(plate, cmap='gray')
            plt.title("原始图像")
            plt.axis('off')

        # 1.统计每行的颜色跳变次数
        change_counts = []
        for i in range(rows):
            change_count = 0
            for j in range(cols - 1):
                pixel_front = plate[i, j]
                pixel_back = plate[i, j + 1]
                if pixel_front != pixel_back:
                    change_count += 1
            change_counts.append(change_count)

        if self.debug:
            # 显示颜色跳变次数分布
            plt.subplot(132)
            plt.plot(change_counts, range(rows))
            plt.gca().invert_yaxis()  # 翻转y轴使其与图像方向一致
            plt.axvline(x=self.min_change_count, color='r', linestyle='--',
                        label=f'最小跳变次数={self.min_change_count}')
            plt.axvline(x=self.max_change_count, color='g', linestyle='--',
                        label=f'最大跳变次数={self.max_change_count}')
            plt.legend()
            plt.title("颜色跳变次数分布")
            plt.xlabel("跳变次数")
            plt.ylabel("行号")

        # 2.计算符合字符特征的行数（字符高度）
        char_height = sum(1 for count in change_counts
                          if self.min_change_count <= count <= self.max_change_count)

        # 4.清除铆钉所在行（颜色跳变次数小于最小值的行）
        for i in range(rows):
            if change_counts[i] < self.min_change_count:
                plate[i, :] = 0

        if self.debug:
            plt.subplot(133)
            plt.imshow(plate, cmap='gray')
            plt.title(f"去除柳丁后\n有效字符高度:{char_height}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # 打印统计信息
            print("\n柳丁去除统计信息:")
            print(f"图像高度: {rows}行")
            print(f"有效字符高度: {char_height}行")
            print(f"最大跳变次数: {max(change_counts)}")
            print(f"最小跳变次数: {min(change_counts)}")
            print(f"平均跳变次数: {sum(change_counts) / len(change_counts):.2f}")
            print(f"被识别为柳丁的行数: {sum(1 for count in change_counts if count < self.min_change_count)}")

        return plate

    def _remove_border(self, plate: np.ndarray) -> Optional[np.ndarray]:
        """
        通过分析垂直方向的颜色跳变次数清除车牌两侧的边框
        Args:
            plate: 输入的二值化车牌图像(已去除柳丁)
        Returns:
            处理后的图像，处理失败返回None
        """
        if plate is None:
            return None

        plate = plate.copy()
        rows, cols = plate.shape
        border_width = 6  # 边框检测范围

        if self.debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(plate, cmap='gray')
            plt.title("去除柳丁后图像")
            plt.axis('off')

        # 1.统计左右两边垂直方向的颜色跳变次数
        left_changes = []
        right_changes = []

        # 检测左边界
        for j in range(border_width):
            change_count = 0
            for i in range(rows - 1):
                pixel_up = plate[i, j]
                pixel_down = plate[i + 1, j]
                if pixel_up != pixel_down:
                    change_count += 1
            left_changes.append(change_count)

        # 检测右边界
        for j in range(cols - border_width, cols):
            change_count = 0
            for i in range(rows - 1):
                pixel_up = plate[i, j]
                pixel_down = plate[i + 1, j]
                if pixel_up != pixel_down:
                    change_count += 1
            right_changes.append(change_count)

        if self.debug:
            # 显示垂直方向跳变次数分布
            plt.subplot(132)
            plt.plot(range(border_width), left_changes, 'r-', label='左边界')
            plt.plot(range(cols - border_width, cols), right_changes, 'b-', label='右边界')
            plt.axhline(y=5, color='g', linestyle='--', label='跳变阈值')
            plt.legend()
            plt.title("垂直方向跳变分布")
            plt.xlabel("列号")
            plt.ylabel("跳变次数")

        # 2.清除满足条件的边框列
        # 这里使用更严格的条件：跳变次数大于8次视为边框
        threshold = 4

        # 处理左边界
        for j in range(border_width):
            if left_changes[j] < threshold:
                plate[:, j] = 0

        # 处理右边界
        for j, col in enumerate(range(cols - border_width, cols)):
            if right_changes[j] < threshold:
                plate[:, col] = 0

        if self.debug:
            plt.subplot(133)
            plt.imshow(plate, cmap='gray')
            plt.title("去除边框后")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            # 打印统计信息
            print("\n边框去除统计信息:")
            print(f"左边界最大跳变次数: {max(left_changes)}")
            print(f"右边界最大跳变次数: {max(right_changes)}")
            print(f"左边界被移除的列数: {sum(1 for count in left_changes if count > threshold)}")
            print(f"右边界被移除的列数: {sum(1 for count in right_changes if count > threshold)}")

        return plate

    def refine(self, plate_img: np.ndarray) -> Optional[np.ndarray]:
        if plate_img is None:
            return None

        try:
            # 保存原始尺寸用于后续验证
            orig_h, orig_w = plate_img.shape[:2]

            # 1. 图像预处理
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 2. 边缘检测
            edges = cv2.Canny(blur, 50, 150)

            # 3. 形态学处理
            kernel_x_large = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 15))
            image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_x_large, iterations=1)

            final_kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            final_kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

            # X方向闭操作
            image = cv2.dilate(image, final_kernel_x)
            image = cv2.erode(image, final_kernel_x)

            # Y方向开操作
            image = cv2.erode(image, final_kernel_y)
            image = cv2.dilate(image, final_kernel_y)

            # 4. 中值滤波去噪
            image = cv2.medianBlur(image, 9)
            filtered_image = image.copy()

            if self.debug:
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(edges, cmap='gray')
                plt.title("边缘检测")
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(image, cmap='gray')
                plt.title("形态学处理")
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(filtered_image, cmap='gray')
                plt.title("中值滤波")
                plt.axis('off')
                plt.tight_layout()
                plt.show()


            # 5. 轮廓处理和边界确定
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return plate_img

            max_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            if self.debug:
                debug_img = plate_img.copy()
                cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
                plt.figure(figsize=(8, 4))
                plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
                plt.title("检测到的边界")
                plt.axis('off')
                plt.show()

            # 6. 计算边界
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(plate_img.shape[1], x_max)
            y_max = min(plate_img.shape[0], y_max)

            # 7. 裁剪车牌
            plate_region = plate_img[y_min:y_max, x_min:x_max]

            # 8. 对裁剪区域进行二值化处理
            plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 计算白色像素比例
            white_ratio = np.sum(binary == 255) / (binary.shape[0] * binary.shape[1])
            print(f"检测到白色像素比例为{white_ratio:.2f}")

            # 如果白色像素比例超过50%，进行翻转
            if white_ratio > 0.4:
                binary = cv2.bitwise_not(binary)
                print(f"检测到白色像素比例为{white_ratio:.2f}，已执行颜色翻转")

            # 9. 去除柳丁和边框
            rivet_removed = self._remove_rivets(binary)
            border_removed = self._remove_border(rivet_removed)


            if self.debug:
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
                plt.title("裁剪后的车牌")
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(binary, cmap='gray')
                plt.title("二值化结果")
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(border_removed, cmap='gray')
                plt.title("去除柳丁")
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                # 输出尺寸信息
                new_w = x_max - x_min
                new_h = y_max - y_min
                print("\n尺寸变化信息:")
                print(f"原始尺寸: {orig_w}x{orig_h}")
                print(f"裁剪后尺寸: {new_w}x{new_h}")
                print(f"宽度变化率: {new_w / orig_w:.2f}")
                print(f"高度变化率: {new_h / orig_h:.2f}")

            return border_removed

        except Exception as e:
            print(f"精确定位过程出现错误: {str(e)}")
            return plate_img