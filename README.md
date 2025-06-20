# PyTorch 水果蔬菜分类预测工具

这是一个基于 PyTorch 的预训练模型，可以识别和分类水果和蔬菜图像。模型已经训练完成，可以直接用于图像预测。

## 项目结构

```
pytorch-computer-vision/
├── src/                   # 源代码
│   ├── model.py           # 神经网络模型定义
│   ├── predict.py         # 图像预测脚本
│   ├── gui_predict.py     # GUI 预测界面
│   ├── gui_results.py     # GUI 结果显示
│   ├── model.pth          # 预训练模型文件
│   └── utils.py           # 工具函数
├── pyproject.toml         # 项目配置和依赖
├── .gitignore             # Git 忽略文件
└── README.md              # 项目说明
```

## 功能特性

- 🍎 支持 36 种水果和蔬菜分类
- 🧠 基于深度学习的图像识别
- 🎯 高精度的分类预测
- 🖥️ 图形用户界面
- ⚡ 即开即用，无需训练

## 快速开始

### 1. 安装依赖

```
uv sync
```

### 2. 准备图片

将您要预测的图片放在任意位置，支持常见图片格式（jpg, png, jpeg 等）。

### 3. 开始预测

#### 命令行方式：
```
# 预测单张图像，显示前 5 个可能的结果
uv run python src/predict.py --image <图片路径> --model src/model.pth --top_k 5

# 示例：
uv run python src/predict.py --image my_apple.jpg --model src/model.pth --top_k 3
```

#### GUI 方式：
```
uv run python src/gui_predict.py
```

然后在图形界面中选择图片文件即可。

## 支持的类别

项目支持识别以下 36 种水果和蔬菜：

**水果 (13 种)：**
- 苹果 (apple)、香蕉 (banana)、葡萄 (grapes)、奇异果 (kiwi)
- 柠檬 (lemon)、芒果 (mango)、橙子 (orange)、梨 (pear)
- 菠萝 (pineapple)、石榴 (pomegranate)、西瓜 (watermelon)

**蔬菜 (23 种)：**
- 甜菜根 (beetroot)、甜椒 (bell pepper)、卷心菜 (cabbage)、辣椒 (capsicum)
- 胡萝卜 (carrot)、花椰菜 (cauliflower)、辣椒 (chilli pepper)、玉米 (corn)
- 黄瓜 (cucumber)、茄子 (eggplant)、大蒜 (garlic)、生姜 (ginger)
- 韭菜 (jalepeno)、生菜 (lettuce)、洋葱 (onion)、辣椒粉 (paprika)
- 豌豆 (peas)、土豆 (potato)、萝卜 (raddish)、大豆 (soy beans)
- 菠菜 (spinach)、甜玉米 (sweetcorn)、红薯 (sweetpotato)、番茄 (tomato)、萝卜 (turnip)

## 使用示例

### 命令行预测结果示例：
```
图像: my_apple.jpg
预测结果:
1. apple: 95.67%
2. pear: 3.21%
3. pomegranate: 1.12%
```

## 模型信息

- **模型架构**: 基于卷积神经网络 (CNN)
- **输入尺寸**: 224x224 像素
- **输出**: 36 个类别的概率分布
- **预处理**: 自动调整图片尺寸和标准化

## 技术要求

- Python 3.11
- uv (Python 包管理器)

## 注意事项

- 确保输入图片清晰，主体明确
- 支持常见图片格式：jpg, jpeg, png, bmp
- 模型文件 (`src/model.pth`) 已包含在项目中
- 无需额外下载或训练模型
