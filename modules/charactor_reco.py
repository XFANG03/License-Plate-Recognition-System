import cv2
import numpy as np
import os
from typing import List, Optional, Tuple, Dict
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt


class PlateType(Enum):
    """车牌类型枚举"""
    MAINLAND = "mainland"  # 大陆车牌
    HK_MC = "hongkong or macau"  # 香港车牌
    UNKNOWN = "unknown"  # 未知类型


@dataclass
class RecognitionResult:
    """识别结果数据类"""
    text: str  # 识别出的文字
    plate_type: PlateType  # 车牌类型
    confidence: float  # 置信度
    debug_images: List[np.ndarray] = None  # 调试图像


class PlateConfig:
    """配置类"""

    def __init__(self):
        self.DEBUG = True
        self.MIN_CONFIDENCE = 0.6  # 最小置信度阈值

        # 模板匹配参数
        self.TEMPLATE_MATCH_METHOD = cv2.TM_CCOEFF_NORMED
        self.TEMPLATE_SCALE_FACTOR = 1.0

        # 字符识别参数
        self.CHAR_SIZE = (20, 40)  # 标准化字符大小
        self.CHAR_MARGIN = 2  # 字符边距


class EnhancedTemplateMatcher:
    """增强型模板匹配器"""

    def __init__(self, template_dir: str, config: Optional[PlateConfig] = None):
        """
        初始化模板匹配器
        Args:
            template_dir: 模板目录路径
            config: 配置对象
        """
        self.template_dir = template_dir
        self.config = config or PlateConfig()

        # 初始化模板字符映射
        self.template = {
            'provinces': ['京', '津', '冀', '晋', '蒙', '辽', '吉', '黑', '沪',
                          '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘',
                          '粤', '桂', '琼', '川', '贵', '云', '渝', '藏', '陕',
                          '甘', '青', '宁', '新'],
            'special': ['港', '澳'],
            'alphabets': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                          'W', 'X', 'Y', 'Z','`'],
            'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        }

        # 加载模板
        self._load_templates()

        # 调试图像列表
        self.debug_images = []

    def _load_templates(self):
        """加载所有模板"""
        self.province_templates = self._load_template_group('provinces')
        self.alphabet_templates = self._load_template_group('alphabets')
        self.number_templates = self._load_template_group('numbers')
        self.special_templates = self._load_template_group('special')

        if self.config.DEBUG:
            print("模板加载统计:")
            print(f"省份字符: {len(self.province_templates)} 组")
            print(f"字母字符: {len(self.alphabet_templates)} 组")
            print(f"数字字符: {len(self.number_templates)} 组")
            print(f"特殊字符: {len(self.special_templates)} 组")

    def _load_template_group(self, group: str) -> Dict[str, List[str]]:
        """
        加载指定组的模板
        Args:
            group: 模板组名称
        Returns:
            模板路径字典
        """
        templates = {}
        base_dir = os.path.join(self.template_dir, group)

        if not os.path.exists(base_dir):
            if self.config.DEBUG:
                print(f"警告: 模板目录不存在 - {base_dir}")
                print("请确保目录结构正确")
            return templates

        for char in self.template[group]:
            char_dir = os.path.join(base_dir, str(char))
            if not os.path.exists(char_dir):
                if self.config.DEBUG:
                    print(f"警告: 字符目录不存在 - {char_dir}")
                continue

            valid_files = []
            for f in os.listdir(char_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(char_dir, f)
                    if os.path.exists(file_path):
                        valid_files.append(file_path)

            if valid_files:
                templates[char] = valid_files
            elif self.config.DEBUG:
                print(f"警告: 未找到有效的模板文件 - {char_dir}")

        return templates

    def _add_debug_image(self, image: np.ndarray, title: str):
        """添加调试图像"""
        if self.config.DEBUG:
            plt.rcParams['font.size'] = 20  # 设置全局字体大小
            self.debug_images.append({
                'image': image.copy(),
                'title': title
            })

    def _show_debug_images(self):
        """显示调试图像"""
        if not self.config.DEBUG or not self.debug_images:
            return

        num_images = len(self.debug_images)
        cols = min(5, num_images)
        rows = (num_images - 1) // cols + 1

        # 直接设置缩放比例为0.5
        scale_factor = 0.5

        plt.figure(figsize=(4 * cols * scale_factor, 4 * rows * scale_factor))

        for idx, debug_info in enumerate(self.debug_images):
            plt.subplot(rows, cols, idx + 1)

            # 获取原始图像并调整大小
            image = debug_info['image']
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height))

            # 显示图像
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            plt.title(debug_info['title'], fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 清空调试图像
        self.debug_images = []

    def _determine_plate_type(self, chars: List[np.ndarray]) -> PlateType:
        """
        根据字符数量和特征确定车牌类型
        Args:
            chars: 字符图像列表
        Returns:
            车牌类型
        """
        if len(chars) < 3:
            return PlateType.UNKNOWN
        elif len(chars) < 7:
            return PlateType.HK_MC
        else:
            return PlateType.MAINLAND

    def _match_mainland_plate(self, chars: List[np.ndarray]) -> RecognitionResult:
        """
        处理大陆车牌
        Args:
            chars: 字符图像列表
        Returns:
            识别结果
        """
        result = []
        confidence_sum = 0

        # 第一位：省份汉字
        char, conf = self._match_character(chars[0], 'provinces')
        result.append(char)
        confidence_sum += conf

        # 第二位：字母
        char, conf = self._match_character(chars[1], 'alphabets')
        result.append(char)
        confidence_sum += conf

        # 检查是否为粤Z车牌
        is_special = (result[0] == '粤' and result[1] == 'Z')

        # 处理剩余字符
        for i, char_img in enumerate(chars[2:], 2):
            if is_special and i == len(chars) - 1:
                # 最后一位可能是"港"或"澳"
                char, conf = self._match_character(char_img, 'special')
            else:
                # 普通字母数字识别
                char, conf = self._match_character(char_img, 'alphanumeric')
            result.append(char)
            confidence_sum += conf

        avg_confidence = confidence_sum / len(chars)
        return RecognitionResult(
            text=''.join(result),
            plate_type=PlateType.MAINLAND,
            confidence=avg_confidence,
            debug_images=self.debug_images
        )

    def _match_hk_plate(self, chars: List[np.ndarray]) -> RecognitionResult:
        """
        处理香港/澳门车牌
        Args:
            chars: 字符图像列表
        Returns:
            识别结果
        """
        result = []
        confidence_sum = 0

        # 前两位必须是字母
        for i in range(2):
            char, conf = self._match_character(chars[i], 'alphabets')
            result.append(char)
            confidence_sum += conf

        # 剩余位数字
        for char_img in chars[2:]:
            char, conf = self._match_character(char_img, 'numbers')
            result.append(char)
            confidence_sum += conf

        avg_confidence = confidence_sum / len(chars)
        return RecognitionResult(
            text=''.join(result),
            plate_type=PlateType.HK_MC,
            confidence=avg_confidence,
            debug_images=self.debug_images
        )

    def _match_character(self, char_img: np.ndarray, char_type: str) -> Tuple[str, float]:
        """
        匹配单个字符
        Args:
            char_img: 字符图像
            char_type: 字符类型
        Returns:
            (匹配的字符, 置信度)
        """
        # 预处理字符图像
        char_img = cv2.resize(char_img, self.config.CHAR_SIZE)

        best_match = '?'
        best_score = float('-inf')

        template_mapping = {
            'provinces': self.province_templates,
            'alphabets': self.alphabet_templates,
            'numbers': self.number_templates,
            'special': self.special_templates,
            'alphanumeric': {**self.alphabet_templates, **self.number_templates}
        }
        templates_to_check = template_mapping.get(char_type, {})

        for char, template_paths in templates_to_check.items():
            for template_path in template_paths:
                template = self._prepare_template(template_path, char_img.shape)
                if template is None:
                    continue

                score = cv2.matchTemplate(
                    char_img,
                    template,
                    self.config.TEMPLATE_MATCH_METHOD
                )[0][0]

                if score > best_score:
                    best_score = score
                    best_match = char

                    if self.config.DEBUG:
                        debug_img = np.hstack([char_img, template])
                        self._add_debug_image(debug_img, f"匹配:{char}={score:.2f}")

        return best_match, best_score


    def recognize(self, chars: List[np.ndarray]) -> RecognitionResult:
        """
        主识别函数
        Args:
            chars: 字符图像列表
        Returns:
            识别结果
        """
        # 清空之前的调试图像
        self.debug_images = []

        # 确定车牌类型
        plate_type = self._determine_plate_type(chars)

        # 根据类型进行识别
        if plate_type == PlateType.MAINLAND:
            result = self._match_mainland_plate(chars)
        elif plate_type == PlateType.HK_MC:
            result = self._match_hk_plate(chars)
        else:
            result = RecognitionResult('', PlateType.UNKNOWN, 0.0, [])

        # 显示调试信息
        if self.config.DEBUG:
            self._show_debug_images()

        return result

    def _prepare_template(self, template_path: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        准备模板图像
        Args:
            template_path: 模板图像路径
            target_shape: 目标尺寸
        Returns:
            处理后的模板图像
        """
        try:
            # 使用 imdecode 替代 imread
            template = cv2.imdecode(
                np.fromfile(template_path, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )

            if template is None:
                if self.config.DEBUG:
                    print(f"无法读取模板: {template_path}")
                return None

            # 调整大小
            template = cv2.resize(template,
                                  (target_shape[1], target_shape[0]))

            # 二值化
            _, template = cv2.threshold(template, 0, 255, cv2.THRESH_OTSU)

            return template

        except Exception as e:
            if self.config.DEBUG:
                print(f"模板处理错误 {template_path}: {str(e)}")
            return None

