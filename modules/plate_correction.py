import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class PlateCorrector:
    def __init__(self):
        pass

    def _show_image(self, title: str, image: np.ndarray):
        """使用matplotlib显示图像"""
        if image is None:
            return
        # 转换为RGB格式以适应matplotlib的显示
        if len(image.shape) == 3:  # 彩色图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        plt.figure(figsize=(6,4))
        plt.title(title)
        plt.imshow(image_rgb, cmap='gray' if len(image.shape) == 2 else None)
        plt.axis('off')
        plt.show()

    def _calculate_skew_angle(self, edges: np.ndarray, plate_img: np.ndarray):
        """计算倾斜角度（针对近水平线条），并返回检测到的水平线条列表"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return None, []

        angles = []
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # 避免除以零
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                if abs(angle) < 45:  # 只考虑水平方向的线段
                    angles.append(angle)
                    horizontal_lines.append((x1, y1, x2, y2))

        if not angles:
            return None, horizontal_lines

        return np.median(angles), horizontal_lines

    def _calculate_background_color(self, image: np.ndarray, margin: int = 5) -> List[int]:
        """计算图像边缘区域的平均颜色作为背景填充色"""
        if len(image.shape) != 3:
            return [0, 0, 0]

        image = image.astype(np.uint8)
        h, w = image.shape[:2]

        top = image[:margin, :]
        bottom = image[-margin:, :]
        left = image[:, :margin]
        right = image[:, -margin:]

        edges = np.vstack([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ])

        mean_color = np.mean(edges, axis=0).round().astype(int)
        return [int(x) for x in mean_color]

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """使用平均背景色旋转图像"""
        image = image.astype(np.uint8)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        bg_color = self._calculate_background_color(image)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color
        ).astype(np.uint8)

        mask = np.ones((height, width), dtype=np.uint8) * 255
        rotated_mask = cv2.warpAffine(
            mask,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        kernel_size = (5, 5)
        blurred_edges = cv2.GaussianBlur(rotated_mask.astype(float) / 255, kernel_size, 0)
        blurred_edges = np.stack([blurred_edges] * 3, axis=2)

        background = np.ones_like(rotated) * np.array(bg_color, dtype=np.uint8)
        result = (rotated * blurred_edges + background * (1 - blurred_edges)).astype(np.uint8)

        return result

    def correct(self, plate_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """车牌校正主函数（水平方向）"""
        if plate_img is None:
            return None, 0

        edges = cv2.Canny(plate_img, 50, 150, apertureSize=3)
        angle, horizontal_lines = self._calculate_skew_angle(edges, plate_img)

        # 可视化校正前后的结果
        if horizontal_lines:
            # 绘制校正前的水平线条（红色）
            before_img = plate_img.copy()
            for (x1, y1, x2, y2) in horizontal_lines:
                cv2.line(before_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if angle is None:
            return plate_img, 0

        corrected = self._rotate_image(plate_img, angle)

        # 再次检测水平线条
        edges_corrected = cv2.Canny(corrected, 50, 150, apertureSize=3)
        _, horizontal_lines_after = self._calculate_skew_angle(edges_corrected, corrected)

        # 绘制校正后的水平线条（绿色）
        after_img = corrected.copy()
        if horizontal_lines_after:
            for (x1, y1, x2, y2) in horizontal_lines_after:
                cv2.line(after_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 创建一个图像窗口显示校正前后的对比
        plt.figure(figsize=(12, 5))

        # 左侧显示校正前的图像
        plt.subplot(121)
        plt.title("检测到水平方向直线")
        plt.imshow(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 右侧显示校正后的图像
        plt.subplot(122)
        plt.title("水平方向修正后")
        plt.imshow(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return corrected, angle

    def _calculate_vertical_skew_angle(self, edges: np.ndarray, plate_img: np.ndarray):
        """改进的垂直线检测"""
        # 调整霍夫变换参数
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,  # 降低阈值
            minLineLength=30,  # 减小最小线段长度
            maxLineGap=20  # 增加最大间隙
        )

        if lines is None:
            return None, []

        angles = []
        vertical_lines = []

        # 改进的角度筛选
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1e-6:  # 处理完全垂直的情况
                angle = 90.0
            else:
                angle = np.degrees(np.arctan(dy / dx))

            # 放宽垂直线的角度范围
            if abs(angle) > 60:  # 修改为60度而不是45度
                angles.append(angle)
                vertical_lines.append((x1, y1, x2, y2))

        if not angles:
            return None, vertical_lines

        return np.median(angles), vertical_lines

    def correct_vertical(self, plate_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用仿射变换校正垂直线条，保持水平线不变
        """
        if plate_img is None:
            return None, 0

        height, width = plate_img.shape[:2]
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150, apertureSize=3)

        # 1. 检测线条

        # 调整的参数
        lines = cv2.HoughLinesP(
            edges,
            rho=1,  # 距离分辨率（像素）
            theta=np.pi / 180,  # 角度分辨率（弧度）
            threshold=30,  # 降低阈值，使检测更敏感
            minLineLength=20,  # 减小最小线长，因为垂直线可能较短
            maxLineGap=3  # 减小最大间隙，避免错误连接
        )
        if lines is None:
            return plate_img, 0

            # 可视化所有检测到的线条
        debug_img = plate_img.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 使用绿色显示所有检测到的线
        self._show_image("All Detected Lines", debug_img)

        # 2. 分类线条并计算倾斜角度
        vertical_lines = []
        vertical_angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 5:  # 完全垂直
                angle = 90
            else:
                angle = np.degrees(np.arctan2(dy, dx))

            if abs(angle) > 60:  # 接近垂直的线
                vertical_lines.append(line[0])
                vertical_angles.append(angle)

        if not vertical_lines:
            return plate_img, 0

        # 3. 计算平均角度和需要的倾斜调整
        avg_angle = np.mean(vertical_angles)
        shear_angle = 90 - abs(avg_angle)  # 计算需要的剪切角度

        # 4. 构建剪切变换矩阵
        shear_factor = -np.tan(np.radians(shear_angle))
        if avg_angle < 0:
            shear_factor = -shear_factor

        shear_matrix = np.float32([
            [1, shear_factor, 0],
            [0, 1, 0]
        ])

        # 5. 应用剪切变换
        corrected = cv2.warpAffine(
            plate_img,
            shear_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 6. 调试显示
        if True:  # DEBUG模式
            debug_img = plate_img.copy()
            # 绘制垂直线（红色）
            for line in vertical_lines:
                x1, y1, x2, y2 = line
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 创建一个图像，包含两个子图
            plt.figure(figsize=(12, 5))

            # 左侧显示线条检测结果
            plt.subplot(121)
            plt.title("检测到的垂直方向直线")
            plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            # 右侧显示校正后的图像
            plt.subplot(122)
            plt.title("垂直方向修正后")
            plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            print(f"Average vertical angle: {avg_angle:.2f}")
            print(f"Applied shear angle: {shear_angle:.2f}")

        return corrected, shear_angle


