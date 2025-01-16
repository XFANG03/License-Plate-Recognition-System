import cv2
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import os
from config import PlateConfig
from utils import ImageUtils
from modules.color_detection import (ColorRange, ColorMaskProcessor, CandidateEvaluator)
from modules.plate_location import PlateLocator, ImagePreprocessor
from modules.plate_correction import PlateCorrector
from modules.process_dual_plate import PlateSplitter
from modules.plate_refine import PlateRefiner
from modules.charactor_segment import CharacterSegmenter
from modules.charactor_reco import EnhancedTemplateMatcher


class PlateDetectionSystem:
    def __init__(self):
        # 初始化配置
        self.config = PlateConfig()
        self.utils = ImageUtils()
        # 设置matplotlib中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 颜色检测模块
        self.evaluator = CandidateEvaluator(self.config)
        self.processors = {
            'blue': ColorMaskProcessor(ColorRange(*self.config.COLOR_RANGES['blue'])),
            'yellow': ColorMaskProcessor(ColorRange(*self.config.COLOR_RANGES['yellow'])),
            'green': ColorMaskProcessor(ColorRange(*self.config.COLOR_RANGES['green'])),
            'white': ColorMaskProcessor(ColorRange(*self.config.COLOR_RANGES['white']))
        }

        # 形态学处理模块
        self.preprocessor = ImagePreprocessor()
        self.locator = PlateLocator()

        # 校正和优化模块
        self.corrector = PlateCorrector()
        self.splitter = PlateSplitter()
        self.refiner = PlateRefiner()

        # 字符处理模块
        self.char_segmenter = CharacterSegmenter()
        self.char_recognizer = EnhancedTemplateMatcher(
            template_dir=r"C:\Users\Lenovo\Desktop\digital_process_Assignment\templates",
            config=self.config
        )

    def detect_by_color(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """使用颜色识别方法定位车牌"""
        # 生成所有颜色的掩码
        masks = {color: processor.process(image)
                 for color, processor in self.processors.items()}

        # 评估所有颜色的候选区域
        candidates_by_color = self.evaluator.evaluate_all_candidates(masks, image)

        # 选择最佳候选区域
        best_candidate = self.evaluator.select_best_candidate(candidates_by_color)

        if best_candidate:
            # 提取车牌图像
            x, y, w, h = cv2.boundingRect(best_candidate.box)
            plate_img = image[y:y + h, x:x + w]
            return plate_img, best_candidate.color

        return None, ""

    def detect_by_morphology(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用形态学方法定位车牌"""
        processed = self.preprocessor.process(image)
        plates = self.locator.locate(processed, image)
        return plates

    def correct_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """车牌矫正处理，包含水平和垂直方向校正"""
        if plate_img is None:
            return None

        # 首先进行水平方向校正
        corrected_plate, angle_h = self.corrector.correct(plate_img)

        if self.config.DEBUG:
            print(f"水平方向校正角度: {angle_h:.2f}°")
            if abs(angle_h) > 1:
                print("已进行水平方向校正")
            else:
                print("水平方向无需校正")

        # 然后进行垂直方向校正
        corrected_plate, angle_v = self.corrector.correct_vertical(corrected_plate)

        if self.config.DEBUG:
            print(f"垂直方向校正角度: {angle_v:.2f}°")
            if abs(angle_v) > 1:
                print("已进行垂直方向校正")
            else:
                print("垂直方向无需校正")

        return corrected_plate

    def segment_characters(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """
        分割车牌字符
        Args:
            plate_img: 车牌图像
        Returns:
            分割后的字符图像列表
        """
        if plate_img is None:
            return []

        try:
            print("\n2. 执行字符分割...")
            char_images = self.char_segmenter.segment(plate_img)

            if self.config.DEBUG:
                print(f"成功分割出 {len(char_images)} 个字符")

            return char_images

        except Exception as e:
            print(f"字符分割过程出现错误: {str(e)}")
            return []

    def recognize_characters(self, char_images: List[np.ndarray]) -> str:
        """
        使用增强版字符识别系统识别字符
        Args:
            char_images: 字符图像列表
        Returns:
            识别结果字符串
        """
        if not char_images:
            return ""

        try:
            print("\n3. 执行字符识别...")
            # 使用新的识别方法
            result = self.char_recognizer.recognize(char_images)

            if self.config.DEBUG:
                print(f"识别结果: {result.text}")
                print(f"车牌类型: {result.plate_type.value}")
                print(f"置信度: {result.confidence:.2f}")

                if result.confidence < self.config.MIN_CONFIDENCE:
                    print("警告: 置信度较低，建议人工复查")

            return result.text

        except Exception as e:
            print(f"字符识别过程出现错误: {str(e)}")
            return ""

    def detect(self, image_path: str) -> Optional[List[Tuple[np.ndarray, str]]]:

        print("\n=== 开始车牌检测 ===")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        results = []

        # 1. 首先尝试颜色识别方法
        print("\n1. 尝试颜色识别方法...")
        plate_img, plate_color = self.detect_by_color(image)

        # 2. 根据颜色识别结果决定后续处理
        print(f"检测到{plate_color}色车牌")
        # 如果是黄色或白色车牌，使用形态学方法进行精确定位
        if plate_color in ['white']:
            print("使用形态学方法进行精确定位...")
            morphology_result = self.detect_by_morphology(image)
            print("形态学定位成功")
            print("车牌矫正开始")
            corrected_morphology = self.correct_plate(morphology_result)
            print("分析是否为双层车牌")
            plates = self.splitter.process(corrected_morphology)
            for plate in plates:
                print("========车牌精细矫正开始===========")
                refined_plate= self.refiner.refine(plate)
                if refined_plate is not None:
                    # 进行字符分割和识别
                    char_images = self.segment_characters(refined_plate)
                    print(111111111)
                    plate_number = self.recognize_characters(char_images)
                    results.append((refined_plate, plate_number))
        else:
            print("使用颜色识别结果")
            corrected_plate = self.correct_plate(plate_img)
            print("车牌精细矫正开始")
            refined_plate = self.refiner.refine(corrected_plate)
            char_images = self.segment_characters(refined_plate)
            plate_number = self.recognize_characters(char_images)
            results.append((refined_plate, plate_number))

        # 3. 如果颜色识别失败，尝试形态学方法
        if not results:
            print("\n2. 颜色识别失败，尝试形态学方法...")
            plate_img = self.detect_by_morphology(image)
            print("形态学定位成功")
            corrected_plate = self.correct_plate(plate_img)
            if corrected_plate is not None:
                print("\n4. 分析是否为双层车牌...")
                plates = self.splitter.process(corrected_plate)
                for plate in plates:
                    print("车牌精细矫正开始")
                    refined_plate = self.refiner.refine(plate)
                    if refined_plate is not None:
                        char_images = self.segment_characters(refined_plate)
                        plate_number = self.recognize_characters(char_images)
                        results.append((refined_plate, plate_number))

        if not results:
            print("未能检测到车牌")
            return None

        return results

    def show_result(self, results: Optional[List[Tuple[np.ndarray, str]]]):
        """
        显示检测结果
        Args:
            results: 车牌图像和识别结果的列表
        """
        if not results:
            print("没有检测到车牌")
            return

        n_plates = len(results)

        fig = plt.figure(figsize=(7, 3))

        for i, (plate, number) in enumerate(results):
            plt.subplot(1, n_plates, i + 1)
            plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))

            # 根据字符数量判断车牌类型
            plate_type = "港澳车牌" if len(number) < 7 else "大陆车牌"
            plt.title(f"车牌 {i + 1} ({plate_type})\n识别结果: {number}")

            # 获取当前轴对象
            ax = plt.gca()

            # 将坐标轴与图像像素对齐
            ax.set_xlim([0, plate.shape[1]])
            ax.set_ylim([plate.shape[0], 0])  # 图像的(0,0)在左上角

            # 隐藏坐标轴刻度和边框
            plt.axis('off')

            # 添加美观的程序化特征：半透明斜体水印文本网格
            watermark_text = "方怡萱20223802012"  # 可根据需要更改文本
            step = 25  # 网格步长，越小越密集
            color = (0.1, 0.1, 0.1)  # 深灰色
            alpha = 0.5  # 半透明度

            # 在图像上铺设斜体文本水印网格
            # 由于图像高宽不大，使用简单循环
            for y_pos in range(0, plate.shape[0], step):
                for x_pos in range(0, plate.shape[1], step):
                    # 使用text函数添加文本，并利用transform和rotation实现倾斜
                    ax.text(x_pos, y_pos, watermark_text,
                            fontsize=8, color=color, alpha=alpha,
                            rotation=30, ha='center', va='center',
                            fontstyle='italic')

        plt.tight_layout()
        plt.show()


def main():
    """测试主函数"""

    # 创建检测系统实例
    system = PlateDetectionSystem()

    # 测试图片
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_images = [
        os.path.join(current_dir, "data", "dual1.jpg"),
    ]

    for image_path in test_images:
        print(f"\n处理图片: {image_path}")

        # 检测车牌
        results = system.detect(image_path)

        # 显示结果
        system.show_result(results)


if __name__ == "__main__":
    main()