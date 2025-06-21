import json
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.model import FruitVegClassifier
from src.utils import get_data_loaders

# 配置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def train_model(data_dir, num_epochs=50, lr=0.0001, weight_decay=1e-5):
    """训练模型并评估每个类别的性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取数据加载器
    train_loader, val_loader, test_loader, classes = get_data_loaders(data_dir)
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别数: {len(classes)}")

    # 初始化模型
    model = FruitVegClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # 记录训练指标
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch + 1}: 保存最佳模型 (验证准确率: {val_acc:.2f}%)")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"训练损失: {avg_train_loss:.4f}, "
            f"验证损失: {val_loss:.4f}, "
            f"验证准确率: {val_acc:.2f}%"
        )

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 评估测试集
    print("\n在测试集上评估模型...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"测试准确率: {test_acc:.2f}%")

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model_{timestamp}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": classes,
            "num_classes": len(classes),
            "training_metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "best_val_accuracy": best_val_acc,
                "test_accuracy": test_acc,
            },
        },
        model_path,
    )
    print(f"模型已保存至: {model_path}")

    # 生成详细评估报告
    print("\n生成详细评估报告...")
    test_results = evaluate_model_detailed(model, test_loader, device, classes)
    val_results = evaluate_model_detailed(model, val_loader, device, classes)

    # 保存评估结果
    results_path = f"evaluation_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "test": test_results,
                "val": val_results,
                "classes": classes,
                "training_metrics": {
                    "train_losses": [float(loss) for loss in train_losses],
                    "val_losses": [float(loss) for loss in val_losses],
                    "val_accuracies": [float(acc) for acc in val_accuracies],
                    "best_val_accuracy": float(best_val_acc),
                    "test_accuracy": float(test_acc),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 绘制训练指标
    plot_training_metrics(
        train_losses, val_losses, val_accuracies, f"training_metrics_{timestamp}.png"
    )

    # 绘制类别准确率
    plot_class_accuracy(test_results, f"test_class_accuracy_{timestamp}.png")
    plot_class_accuracy(val_results, f"val_class_accuracy_{timestamp}.png")

    print(f"评估结果已保存至: {results_path}")
    return model, classes, model_path


def evaluate_model(model, data_loader, criterion, device):
    """评估模型并返回损失和准确率"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate_model_detailed(model, data_loader, device, classes):
    """评估模型并返回每个类别的详细性能指标"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成分类报告
    report = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True
    )

    # 提取每个类别的指标
    class_results = {}
    for class_name in classes:
        if class_name in report:
            class_results[class_name] = {
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }

    # 添加总体指标
    class_results["overall"] = {
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }

    return class_results


def plot_training_metrics(train_losses, val_losses, val_accuracies, filename):
    """绘制训练过程中的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.title("训练和验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="验证准确率", color="red")
    plt.title("验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率 (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)


def plot_class_accuracy(results, filename):
    """绘制每个类别的准确率柱状图"""
    classes = [c for c in results if c != "overall"]
    f1_scores = [results[c]["f1-score"] for c in classes]

    plt.figure(figsize=(12, 6))
    plt.bar(classes, f1_scores, color="skyblue")
    plt.title("各类别 F1 分数")
    plt.xlabel("类别")
    plt.ylabel("F1 分数")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
