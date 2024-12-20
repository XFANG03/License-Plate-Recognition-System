# 智能车牌识别系统

> 基于双通道定位与自适应矫正的智能车牌识别系统

## 📢 项目简介

这是一套基于 OpenCV 的智能车牌识别系统，专门面向复杂道路环境下的中国机动车车牌识别需求。系统采用颜色检测与形态学处理的双通道策略进行车牌定位，并创新性地引入了自适应背景填充技术来优化车牌矩正效果。
代码将于25年1月10日后正式开源，敬请期待！

💡 **如果您认为这个项目有价值，请给我一个 Star，并分享给更多人！🌟**

---

### 主要特性

- 🚗 **双通道车牌定位**

  - 颜色检测通道：支持蓝、黄、绿、白等多色系车牌检测
  - 形态学处理通道：基于边缘特征和几何形态的定位方法
  - 多维度特征评估和优先级排序机制

- 🔧 **自适应车牌矩正**

  - 水平方向旋转与垂直方向剪切变换相结合
  - 创新的自适应背景填充技术
  - 有效降低矩正过程中的背景器静干扰

- ✂️ **精确字符分割**

  - 基于折尖曲线分析的双层车牌分割
  - 自适应阈值处理
  - 铜钉和边框去除优化

- 🔍 **高精度字符识别**

  - 模板匹配与车牌格式验证相结合
  - 特殊字符处理
  - 多类型车牌支持（民用、新能源、港澳车牌等）

---

## 系统架构

项目采用经典的 MVC 架构设计，确保了系统的模块化和高扩展性：

- **Model层**：核心算法实现，包括车牌定位、图像矩正、字符分割与识别等
- **Controller层**：负责模块间的协调和全局流程管理
- **View层**：提供直观的图形化操作界面

<p align="center">
  <img src="https://github.com/user-attachments/assets/15c6408a-a7de-4046-ad74-c40c40760a24" alt="系统架构">
</p>

---

## 环境要求

- Python 3.8+
- OpenCV 4.5+
- NumPy
- Matplotlib
- tkinter (GUI界面)

---

## 安装说明

```bash
# 克隆项目
git clone https://github.com/XFANG03/License-Plate-Recognition-System.git

# 进入项目目录
cd License-Plate-Recognition-System

# 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

1. 运行主程序：

```bash
python index.py
```

2. 在GUI界面中选择要识别的车牌图片

3. 系统将自动进行处理并显示识别结果

---

## 核心模块说明
<p align="center">
  <img src="https://github.com/user-attachments/assets/91414bd7-fdc4-4444-a252-9d639342ac3d" alt="测试场景1" width="300">
</p>


### 1. 车牌定位模块 (`plate_location.py`)

- HSV色彩空间分析
- 多级形态学处理
- 轮尾检测与筛选


### 2. 车牌矩正模块 (`plate_correction.py`)

- Canny边缘检测
- Hough直线检测
- 仰属变换校正


### 3. 车牌分割模块 (`plate_refine.py`)

- 水平折尖分析
- 自适应阈值处理
- 边框和铜钉去除

### 4. 字符分割模块 (`charactor_segment.py`)

- 垂直折尖分析
- 连通区域分析
- 字符提取与验证

### 5. 字符识别模块 (`charactor_reco.py`)

- 模板匹配实现
- 信心度评估
- 格式验证

---

## 测试结果

系统在多个测试场景下表现出色：

- 支持多种类型车牌识别（民用、新能源、港澳车牌）
- 适应复杂环境（光照变化、倾斜角度等）
- 处理效率高，识别准确率高

| 场景        | 测试图片                                               |
|-------------|--------------------------------------------------------|
| 普通车牌     |  <img src="https://github.com/user-attachments/assets/bcbaf923-d28a-4829-ab8a-fff3d92dbb1b" alt="普通车牌" width="300"> |
| 倾斜车牌      |<img src="https://github.com/user-attachments/assets/f233f057-cd3c-4997-acf1-11131009662d" alt="倾斜车牌"  width="300">  |
| 香港车牌      | <img src="https://github.com/user-attachments/assets/7afd083e-a9b1-4d77-bd03-184969a3ba00" alt="香港车牌" width="300">  |
| 双车牌     | <img src="https://github.com/user-attachments/assets/7dc5f12a-68b4-4d87-8137-cf53ae556185" alt="双车牌" width="300"> |



---

## 致谢

感谢所有对本项目提供帮助和建议的人！


