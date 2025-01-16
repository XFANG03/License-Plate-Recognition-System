import cv2
import numpy as np
from typing import List, Optional
from config import PlateConfig
from utils import ImageUtils

class PlateSplitter:
    """车牌分割器：用于检测和分割双层车牌"""

    def __init__(self):
        self.config = PlateConfig()
        self.utils = ImageUtils()

    def process(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """
        处理输入的车牌图像
        Args:
            plate_img: BGR格式的车牌图像
        Returns:
            分割后的车牌图像列表（单车牌返回长度为1的列表）
        """
        if self._is_dual_plate(plate_img):
            print("检测到双层车牌，进行分割...")
            return self._split_plate(plate_img)
        else:
            print("检测到单层车牌")
            return [plate_img]

    def _is_dual_plate(self, plate_img: np.ndarray) -> bool:
        """
        通过水平投影分析判断是否为双层车牌
        Args:
            plate_img: BGR格式的车牌图像
        Returns:
            是否为双层车牌
        """
        # 转为灰度图
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # 自适应二值化，以处理不同光照条件
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 计算水平投影并转换为合适的数据类型
        projection = np.sum(binary, axis=1, dtype=np.float32)  # 使用 float32 类型

        # 归一化投影值到0-255范围
        if projection.max() > 0:  # 避免除以零
            projection = (projection * 255.0 / projection.max()).astype(np.uint8)
        else:
            projection = projection.astype(np.uint8)

        # 使用高斯滤波平滑投影曲线
        smoothed = cv2.GaussianBlur(projection.reshape(-1, 1), (5, 1), 0).flatten()

        # 查找波峰
        peaks = self._find_peaks(smoothed)
        if len(peaks) < 2:
            return False

        # 分析波峰特征
        peak_distances = np.diff(peaks)
        height = plate_img.shape[0]

        # 判断标准：
        # 1. 至少有两个波峰
        # 2. 波峰之间的距离合适（在图像高度的15%-70%之间）
        min_distance = height * 0.15
        max_distance = height * 0.70

        has_valid_peaks = any((min_distance <= dist <= max_distance)
                              for dist in peak_distances)

        return has_valid_peaks

    def _split_plate(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """分割双层车牌"""
        # 转为灰度图
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # 计算并归一化水平投影
        projection = np.sum(binary, axis=1, dtype=np.float32)
        projection = (projection * 255.0 / projection.max()).astype(np.uint8) if projection.max() > 0 else projection

        # 使用高斯滤波平滑投影曲线
        smoothed = cv2.GaussianBlur(projection.reshape(-1, 1), (5, 1), 0).flatten()

        # 使用_find_best_split_point找到分割点
        split_point = self._find_best_split_point(smoothed)

        if split_point is not None:
            # 分割图像
            return [plate_img[:split_point, :], plate_img[split_point:, :]]

        return [plate_img]

    def _find_peaks(self, data: np.ndarray) -> List[int]:
        """
        查找投影曲线中的波峰
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                # 仅考虑显著的波峰
                if (data[i] > np.mean(data) * 1.2):  # 阈值可调
                    peaks.append(i)
        return peaks

    def _find_best_split_point(self, projection: np.ndarray) -> Optional[int]:
        """
        在投影中找到最佳分割点：最深波谷后的第一个显著波峰
        """
        height = len(projection)
        search_start = int(height * 0.3)
        search_end = int(height * 0.7)

        valleys = []
        # 在搜索范围内找到所有波谷
        for i in range(search_start, search_end):
            if (projection[i] < projection[i - 1] and projection[i] < projection[i + 1]):
                left_peak = max(projection[max(0, i - 10):i])
                right_peak = max(projection[i + 1:min(len(projection), i + 11)])
                depth = min(left_peak - projection[i], right_peak - projection[i])
                valleys.append((i, depth))

        if valleys:
            # 找到最深的波谷
            valleys.sort(key=lambda x: x[1], reverse=True)
            deepest_valley = valleys[0][0]

            # 寻找分割点（波峰）
            min_peak_valley_distance = int(height * 0.05)
            max_peak_valley_distance = int(height * 0.2)

            # 在波谷之后的范围内寻找第一个显著波峰
            for i in range(deepest_valley + min_peak_valley_distance,
                           min(deepest_valley + max_peak_valley_distance, len(projection) - 1)):
                if (projection[i] > projection[i - 1] and
                        projection[i] > projection[i + 1] and
                        projection[i] > np.mean(projection[deepest_valley:i + 1]) * 1.1):
                    if self.config.DEBUG:
                        print(f"找到波峰，位置: {i}, 值: {projection[i]}")
                    return i  # 找到波峰后立即返回

            # 如果没找到合适的波峰，返回波谷位置
            if self.config.DEBUG:
                print(f"未找到合适的波峰，使用波谷位置: {deepest_valley}")
            return deepest_valley

        if self.config.DEBUG:
            print("未找到任何波谷")
        return None
