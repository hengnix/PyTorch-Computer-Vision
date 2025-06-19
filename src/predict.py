import torch
from torchvision import transforms
from PIL import Image
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FruitVegClassifier


def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['class_names']
    num_classes = len(classes)

    # 初始化模型
    model = FruitVegClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, classes, device


def predict(image_path, model_path, top_k=5):
    """对单张图像进行预测，返回前k个预测结果"""
    # 加载模型
    model, classes, device = load_model(model_path)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"无法处理图像: {e}")

    # 执行预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, 1)

        # 获取前k个预测结果
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        # 转换为类别名称
        predictions = [
            {
                'class': classes[idx],
                'probability': float(prob),
                'confidence': f"{prob * 100:.2f}%"
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

    return predictions


def main():
    """命令行预测工具主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='水果和蔬菜图像分类预测')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--top_k', type=int, default=5, help='显示前k个预测结果')

    args = parser.parse_args()

    try:
        predictions = predict(args.image, args.model, args.top_k)

        print(f"图像: {args.image}")
        print("预测结果:")
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']}: {pred['confidence']}")

    except Exception as e:
        print(f"预测错误: {e}")


if __name__ == "__main__":
    main()
