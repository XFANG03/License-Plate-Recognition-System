import cv2
class PlateConfig:
    # Debug模式
    DEBUG = True

    # 显示设置
    FONT_SETTINGS = {
        'font.sans-serif': ['SimHei'],
        'axes.unicode_minus': False
    }

    # 颜色检测参数
    COLOR_RANGES = {
        'blue': ([100, 50, 50], [130, 255, 255]),
        'yellow': ([15, 70, 70], [45, 255, 255]),
        'green': ([35, 10, 120], [85, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255])
    }

    # 车牌几何特征参数
    PLATE_FEATURES = {
        'min_aspect_ratio': 2.0,
        'max_aspect_ratio': 5.5,
        'min_rectangularity': 0.5,
        'min_area': 2000,
        'max_area': 30000,
        'ideal_area': 3000
    }

    # 字符分割参数
    CHAR_FEATURES = {
        'width_range': (8, 50),
        'height_range': (20, 80),
        'min_aspect_ratio': 0.2,
        'max_aspect_ratio': 1.0,
        'min_gap': 20  # 最小字符间距
    }

    # 边框检测参数
    BORDER_FEATURES = {
        'min_height_ratio': 0.4,
        'max_height_ratio': 0.9,
        'min_width_ratio': 0.6,
        'max_width_ratio': 0.95,
        'min_continuous_ratio': 0.8
    }
    # 黑白检测参数
    BW_PARAMS = {
        'value_threshold': 80,
        'saturation_threshold': 80,
        'gaussian_kernel': (6, 6),
        'morph_kernel': (17, 5)
    }
    # 添加位置特征参数
    POSITION_FEATURES = {
        'center_x_weight': 0.6,  # 水平中心度权重
        'center_y_weight': 0.4,  # 垂直位置权重
        'ideal_y_ratio': 0.6,  # 理想的垂直位置比例(从上到下)
        'center_tolerance': 0.2  # 中心位置的容差范围
    }
    # 新增字符识别参数
    CHAR_SIZE = (20, 40)  # 标准字符大小 (宽, 高)
    CHAR_MARGIN = 2  # 字符边距
    MIN_CONFIDENCE = 0.6  # 最小置信度阈值

    # 模板匹配参数
    TEMPLATE_MATCH_METHOD = cv2.TM_CCOEFF_NORMED  # 模板匹配方法
    TEMPLATE_SCALE_FACTOR = 1.0  # 模板缩放因子