import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from config import PlateConfig
from utils import ImageUtils


# ================ 数据结构定义 ================
@dataclass
class ColorRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

@dataclass
class PlateCandidate:
    box: np.ndarray
    score: float
    color: str
    features: Dict

# ================ 特征计算类 ================
class FeatureCalculator:
    """负责计算几何特征和位置特征"""

    @staticmethod
    def calculate_geometric_features(contour: np.ndarray, rect) -> Dict:
        """计算几何特征"""
        area = cv2.contourArea(contour)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0

        width, height = rect[1]
        if width < height:
            width, height = height, width
        aspect_ratio = width / height if height > 0 else 0

        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity
        }

    @staticmethod
    def calculate_position_score(box: np.ndarray, image_shape: tuple, config: PlateConfig) -> float:
        """计算位置得分"""
        height, width = image_shape[:2]
        center_x = np.mean(box[:, 0]) / width
        center_y = np.mean(box[:, 1]) / height

        center_x_score = 1 - min(abs(center_x - 0.5) / config.POSITION_FEATURES['center_tolerance'], 1)
        y_diff = abs(center_y - config.POSITION_FEATURES['ideal_y_ratio'])
        center_y_score = 1 - min(y_diff / config.POSITION_FEATURES['center_tolerance'], 1)

        return (center_x_score * config.POSITION_FEATURES['center_x_weight'] +
                center_y_score * config.POSITION_FEATURES['center_y_weight'])


# ================ 掩码处理类 ================
class ColorMaskProcessor:
    def __init__(self, color_range: ColorRange, kernel_size: Tuple[int, int] = (5, 5)):
        self.color_range = color_range
        self.kernel_size = kernel_size

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """形态学处理"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    def process(self, image: np.ndarray) -> np.ndarray:
        """处理车牌"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                          np.array(self.color_range.lower),
                          np.array(self.color_range.upper))
        return self._apply_morphology(mask)

# ================ 评估器类 ================
class CandidateEvaluator:
    def __init__(self, config: PlateConfig):
        self.config = config
        self.feature_calculator = FeatureCalculator()
        # 标准颜色优先级顺序
        self.priority_colors = ['blue', 'green', 'yellow', 'white']

    def evaluate_all_candidates(self, masks: Dict[str, np.ndarray],
                                image: np.ndarray) -> Dict[str, List[PlateCandidate]]:
        """评估所有颜色的候选区域"""
        candidates = {}

        for color, mask in masks.items():
            candidates[color] = self._evaluate_color_mask(mask, image, color)

            if self.config.DEBUG:
                print(f"\n{color} 掩码分析:")
                print(f"检测到的候选区域数: {len(candidates[color])}")

        return {color: cands for color, cands in candidates.items() if cands}

    def _evaluate_color_mask(self, mask: np.ndarray,
                             image: np.ndarray,
                             color: str) -> List[PlateCandidate]:
        """评估单个颜色掩码的所有候选区域"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            candidate = self._evaluate_single_contour(contour, image, color)
            if candidate:
                candidates.append(candidate)

        return sorted(candidates, key=lambda x: x.score, reverse=True)

    def _evaluate_single_contour(self, contour: np.ndarray,
                                 image: np.ndarray,
                                 color: str) -> Optional[PlateCandidate]:
        """评估单个轮廓"""
        features = self._extract_features(contour, image)
        if not features:
            return None

        score = self._calculate_final_score(features)
        return PlateCandidate(
            box=features['box'],
            score=score,
            color=color,
            features=features
        )

    def _extract_features(self, contour: np.ndarray,
                          image: np.ndarray) -> Optional[Dict]:
        """提取并验证特征"""
        rect = cv2.minAreaRect(contour)
        geometric_features = self.feature_calculator.calculate_geometric_features(contour, rect)

        if not self._validate_candidate(geometric_features):
            return None

        box = np.intp(cv2.boxPoints(rect))
        position_score = self.feature_calculator.calculate_position_score(
            box, image.shape, self.config
        )

        features = geometric_features.copy()
        features.update({
            'box': box,
            'position_score': position_score
        })

        return features

    def select_best_candidate(self, candidates: Dict[str, List[PlateCandidate]]) -> Optional[PlateCandidate]:
        """选择最佳候选区域

        首先基于分数选择最佳候选区域，当存在多个相同最高分数的候选区域时，
        按照优先颜色顺序(蓝色、绿色、黄色)进行选择。

        Args:
            candidates: 按颜色分组的候选区域字典
        Returns:
            最佳候选区域，如果没有候选区域则返回None
        """
        if not candidates:
            return None

        # 获取所有候选区域并按分数排序
        all_candidates = [cand for cands in candidates.values() for cand in cands]
        if not all_candidates:
            return None

        # 按分数从高到低排序
        sorted_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
        max_score = sorted_candidates[0].score

        # 获取所有最高分数的候选区域
        best_candidates = [cand for cand in sorted_candidates if abs(cand.score - max_score) < 1e-6]

        if len(best_candidates) == 1:
            return best_candidates[0]

        # 如果有多个最高分候选区域，按照优先颜色顺序选择
        for color in self.priority_colors:
            for candidate in best_candidates:
                if candidate.color == color:
                    return candidate

        # 如果没有优先颜色的候选区域，返回第一个最高分候选区域
        return best_candidates[0]

    def _validate_candidate(self, features: Dict) -> bool:
        """验证候选区域是否满足基本条件"""
        return (
                self.config.PLATE_FEATURES['min_area'] <= features['area'] <=
                self.config.PLATE_FEATURES['max_area'] and
                self.config.PLATE_FEATURES['min_aspect_ratio'] <= features['aspect_ratio'] <=
                self.config.PLATE_FEATURES['max_aspect_ratio'] and
                features['rectangularity'] >= self.config.PLATE_FEATURES['min_rectangularity']
        )

    def _calculate_final_score(self, features: Dict) -> float:
        """计算最终得分"""
        # 基础分数计算
        area_score = self._calculate_area_score(features['area'])
        ratio_score = self._calculate_ratio_score(features['aspect_ratio'])

        base_score = (
                area_score * 0.2 +
                ratio_score * 0.2 +
                features['rectangularity'] * 0.55 +
                features['position_score'] * 0.05
        )

        # 应用加分规则
        return self._apply_bonus_rules(base_score, features)

    def _calculate_area_score(self, area: float) -> float:
        """计算面积得分"""
        return 1.0 - min(
            abs(area - self.config.PLATE_FEATURES['ideal_area']) /
            self.config.PLATE_FEATURES['ideal_area'],
            1.0
        )

    def _calculate_ratio_score(self, aspect_ratio: float) -> float:
        """计算长宽比得分"""
        return 1.0 - min(abs(aspect_ratio - 3.0) / 3.0, 1.0)

    def _apply_bonus_rules(self, base_score: float, features: Dict) -> float:
        """应用加分规则"""
        score = base_score

        if 2.8 <= features['aspect_ratio'] <= 3.2:
            score *= 1.1
        if features['position_score'] > 0.7:
            score *= 1.1
        if features['rectangularity'] > 0.9:
            score *= 1.1

        return score

# ================ 可视化类 ================
class ResultVisualizer:
    def __init__(self,config: PlateConfig):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.config = config
        self.utils = ImageUtils()

    def show_masks(self, image: np.ndarray, masks: Dict[str, np.ndarray],
                   candidates_by_color: Dict[str, List[PlateCandidate]]):
        """显示所有颜色的掩码和检测结果"""
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))

        color_names = {
            'blue': '蓝色掩码',
            'yellow': '黄色掩码',
            'green': '绿色掩码',
            'white': '白色掩码',
        }

        for idx, (color, mask) in enumerate(masks.items()):
            # 显示掩码
            axes[idx, 0].imshow(mask, cmap='gray')
            axes[idx, 0].set_title(color_names.get(color, color))
            axes[idx, 0].axis('off')

            # 显示检测结果
            marked_img = image.copy()
            if color in candidates_by_color:
                candidates = [c.box for c in candidates_by_color[color]]
                cv2.drawContours(marked_img, candidates, -1, (0, 255, 0), 2)

            axes[idx, 1].imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title(f'{color_names.get(color, color)}区域标注')
            axes[idx, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def show_final_results(self, image: np.ndarray, candidates_by_color: Dict[str, List[PlateCandidate]],
                           best_candidate: Optional[PlateCandidate]):
        """显示最终检测结果"""
        result = image.copy()
        color_bgr = {
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'white': (128, 128, 128)
        }

        plt.figure(figsize=(15, 10))

        # 绘制所有候选区域
        for color, candidates in candidates_by_color.items():
            bgr_color = color_bgr[color]
            for candidate in candidates:
                cv2.drawContours(result, [candidate.box], 0, bgr_color, 2)
                x, y = candidate.box[0]
                cv2.putText(result, f"{color}: {candidate.score:.2f}",
                            (int(x), int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

        # 标注最佳候选区域
        if best_candidate:
            cv2.drawContours(result, [best_candidate.box], 0, (0, 0, 255), 3)
            x, y = best_candidate.box[0]
            cv2.putText(result, f"Best: {best_candidate.color} ({best_candidate.score:.2f})",
                        (int(x), int(y) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("车牌检测结果")
        plt.axis('off')
        plt.show()

    def process_image(self, image_path: str, detector: 'LicensePlateDetector') -> None:
        """处理单张图片并可视化结果"""
        image = cv2.imread(image_path)
        if image is None:
            print("无法读取图像")
            return

        # 执行检测
        masks, candidates_by_color, best_candidate = detector.detect(image)

        # 显示中间结果
        self.show_masks(image, masks, candidates_by_color)

        # 显示最终结果
        self.show_final_results(image, candidates_by_color, best_candidate)

        # 打印检测结果
        print("\n检测结果详情:")
        for color, candidates in candidates_by_color.items():
            print(f"\n{color}色候选区域:")
            for i, candidate in enumerate(candidates):
                print(f"  候选区域 {i + 1}: 得分 = {candidate.score:.3f}")
                if self.config.DEBUG:
                    print(f"    位置得分 = {candidate.features['position_score']:.3f}")
                    print(f"    矩形度 = {candidate.features['rectangularity']:.3f}")
                    print(f"    长宽比 = {candidate.features['aspect_ratio']:.3f}")

        if best_candidate:
            print(f"\n最佳候选区域:")
            print(f"颜色: {best_candidate.color}")
            print(f"得分: {best_candidate.score:.3f}")
        else:
            print("\n未检测到有效的车牌区域")

def main():
    """测试颜色识别和车牌候选区域评估功能"""
    # 初始化配置
    config = PlateConfig()

    # 初始化评估器和可视化器
    evaluator = CandidateEvaluator(config)
    visualizer = ResultVisualizer(config)

    # 初始化颜色处理器
    processors = {
        'blue': ColorMaskProcessor(ColorRange(*config.COLOR_RANGES['blue'])),
        'yellow': ColorMaskProcessor(ColorRange(*config.COLOR_RANGES['yellow'])),
        'green': ColorMaskProcessor(ColorRange(*config.COLOR_RANGES['green'])),
        'white': ColorMaskProcessor(ColorRange(*config.COLOR_RANGES['white'])),

    }

    # 测试图像路径列表
    test_images = [
        r"C:\Users\Lenovo\Desktop\digital_process_Assignment\data\27.jpg",
    ]

    for image_path in test_images:
        print(f"\n处理图像: {image_path}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 显示原始图像
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        plt.show()

        # 处理每种颜色
        masks = {}
        print("\n颜色掩码处理结果:")
        for color, processor in processors.items():
            print(f"\n处理 {color} 色:")
            mask = processor.process(image)
            masks[color] = mask

            # 打印掩码统计信息
            white_pixels = np.sum(mask == 255)
            total_pixels = mask.size
            coverage = white_pixels / total_pixels * 100
            print(f"- 掩码覆盖率: {coverage:.2f}%")

        # 评估候选区域
        candidates_by_color = evaluator.evaluate_all_candidates(masks, image)

        # 选择最佳候选区域
        best_candidate = evaluator.select_best_candidate(candidates_by_color)

        # 显示处理结果
        print("\n候选区域评估结果:")
        for color, candidates in candidates_by_color.items():
            print(f"\n{color}色候选区域:")
            for i, candidate in enumerate(candidates):
                print(f"- 候选区域 {i + 1}:")
                print(f"  得分: {candidate.score:.3f}")
                print(f"  面积: {candidate.features['area']:.0f}")
                print(f"  长宽比: {candidate.features['aspect_ratio']:.2f}")
                print(f"  矩形度: {candidate.features['rectangularity']:.2f}")

        if best_candidate:
            print(f"\n最佳候选区域:")
            print(f"- 颜色: {best_candidate.color}")
            print(f"- 得分: {best_candidate.score:.3f}")
        else:
            print("\n未检测到有效的车牌区域")

        # 使用可视化器显示结果
        visualizer.show_masks(image, masks, candidates_by_color)
        visualizer.show_final_results(image, candidates_by_color, best_candidate)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    main()