import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from config import PlateConfig
from utils import ImageUtils


class CharacterSegmenter:
    """车牌字符分割类"""

    def __init__(self, config: PlateConfig = None):
        self.config = config or PlateConfig()
        self.utils = ImageUtils()

    def _dilate_image(self, binary: np.ndarray) -> np.ndarray:
        """
        图像膨胀操作
        Args:
            binary: 二值化图像
        Returns:
            膨胀后的图像
        """
        # 定义膨胀核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,3))

        # 执行膨胀操作
        dilated = cv2.dilate(binary, kernel)

        # 显示原图和膨胀结果在同一张图中
        if self.config.DEBUG:
            # 创建一个 1x2 的子图
            plt.figure(figsize=(10, 5))

            # 显示原图
            plt.subplot(1, 2, 1)
            plt.imshow(binary, cmap='gray')
            plt.title("原图")
            plt.axis('off')

            # 显示膨胀结果
            plt.subplot(1, 2, 2)
            plt.imshow(dilated, cmap='gray')
            plt.title("膨胀结果")
            plt.axis('off')

            # 显示图像

        return dilated

    def _find_character_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        查找并筛选字符轮廓
        Args:
            image: 处理后的图像
        Returns:
            字符轮廓列表 [(x, y, w, h)]
        """
        if self.config.DEBUG:
            print("\n=== 字符轮廓检测开始 ===")
            height, width = image.shape
            print(f"输入图像尺寸: {width}x{height}")

        # 查找轮廓
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.config.DEBUG:
            print(f"\n找到轮廓总数: {len(contours)}")

            # 创建一个彩色图像用于显示轮廓
            debug_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

            # 用不同颜色绘制每个轮廓
            for i, contour in enumerate(contours):
                # 为每个轮廓生成随机颜色
                color = (np.random.randint(0, 255),
                         np.random.randint(0, 255),
                         np.random.randint(0, 255))

                # 绘制轮廓
                cv2.drawContours(debug_image, [contour], -1, color, 2)

                # 获取轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)

            # 显示带有轮廓的图像
            plt.figure(figsize=(10, 6))
            plt.subplot(121)
            plt.imshow(image, cmap='gray')
            plt.title('膨胀后图像')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            plt.title('检测到的所有轮廓')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        # 初始化字符区域列表
        char_regions = []

        # 遍历所有轮廓
        if self.config.DEBUG:
            print("\n开始筛选轮廓:")
            print("序号\t位置(x,y)\t大小(w,h)\t高宽比\t是否通过")

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            ratio = h / w if w > 0 else 0

            # 根据字符的形状特征筛选
            passed = (h > (w * 1)) and (h < (w * 5.5)) and (w > 6) and (h>20)

            if self.config.DEBUG:
                result = "通过" if passed else "未通过"
                area = cv2.contourArea(contour)
                print(f"{i + 1}\t({x},{y})\t({w},{h})\t{ratio:.2f}\t{result}")
                print(f"    轮廓面积: {area:.2f}")

                # 输出未通过的具体原因
                if not passed:
                    if not (h > (w * 1)):
                        print(f"    未通过原因: 高宽比太小 ({ratio:.2f} < 1)")
                    elif not (h < (w * 5.5)):
                        print(f"    未通过原因: 高宽比太大 ({ratio:.2f} > 5.5)")
                    elif not (w > 6):
                        print(f"    未通过原因: 宽度太小 ({w} < 6)")

            if passed:
                char_regions.append((x, y, w, h))

        # 按x坐标排序
        char_regions = sorted(char_regions, key=lambda x: x[0])

        return char_regions

    def _extract_characters(self, image: np.ndarray, char_regions: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        提取单个字符图像
        Args:
            image: 原始图像
            char_regions: 字符区域列表
        Returns:
            字符图像列表
        """
        char_images = []
        image_height, image_width = image.shape[:2]  # 获取图像的高度和宽度

        for x, y, w, h in char_regions:
            # 对字符区域的切割区域进行边界检查
            # 计算切割区域的边界，确保不超出图像范围
            x_start = max(x - 1, 0)  # x - 1 不能小于0
            y_start = max(y - 1, 0)  # y - 1 不能小于0
            x_end = min(x + w + 1, image_width)  # x + w + 1 不能大于图像宽度
            y_end = min(y + h + 1, image_height)  # y + h + 1 不能大于图像高度

            char_img = image[y_start:y_end, x_start:x_end]

            char_images.append(char_img)

        # 显示分割结果
        if self.config.DEBUG and char_images:
            plt.figure(figsize=(15, 3))
            for i, char_img in enumerate(char_images):
                plt.subplot(1, len(char_images), i + 1)
                plt.imshow(char_img, cmap='gray')
                plt.axis('off')
            plt.show()

        return char_images

    def segment(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """
        车牌字符分割主函数
        Args:
            plate_img: 车牌图像
        Returns:
            分割后的字符图像列表
        """
        # 1. 图像膨胀
        dilated = self._dilate_image(plate_img)

        # 2. 查找字符轮廓
        char_regions = self._find_character_contours(dilated)

        if self.config.DEBUG:
            print(f"找到 {len(char_regions)} 个字符区域")
            print("字符位置:", char_regions)

        # 3. 提取字符图像
        char_images = self._extract_characters(plate_img, char_regions)

        return char_images

