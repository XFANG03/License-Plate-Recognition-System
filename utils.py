import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import PlateConfig

class ImageUtils:
    @staticmethod
    def show_image(title: str, image: np.ndarray):
        """显示处理过程的中间结果图像"""
        if PlateConfig.DEBUG:
            plt.rcParams['font.family'] = 'SimHei'
            plt.figure(figsize=(8, 6))
            if len(image.shape) == 2:
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()
