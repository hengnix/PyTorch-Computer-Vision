import warnings

from src.train import train_model

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


if __name__ == "__main__":
    # 数据集路径（需要包含train、val、test三个子文件夹）
    data_dir = "E:/PythonProject/fru_veg_classfier2/dataset"

    # 训练参数
    num_epochs = 50
    learning_rate = 0.0001
    weight_decay = 1e-5

    print(f"开始训练模型，数据集路径: {data_dir}")

    try:
        model, classes, model_path = train_model(
            data_dir=data_dir,
            num_epochs=num_epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        print("\n训练完成!")
        print(f"模型保存路径: {model_path}")
        print(f"识别类别: {classes}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
