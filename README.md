# ResNet-50 图像分类

## 项目简介

本项目基于 **ResNet-50** 深度神经网络模型进行图像分类，使用 **PyTorch** 框架。项目支持图像预处理、数据增强、迁移学习、模型训练与验证、模型保存、推理等功能，适用于图像分类任务。以番茄图像为例，进行多类别的图像分类任务，可以识别不同类型的番茄病害。

项目特点：

- 使用预训练的 **ResNet-50** 模型，进行迁移学习。
- 数据增强（旋转、裁剪、翻转等）用于提高模型鲁棒性。
- 提供提前停止功能，避免过拟合。
- 支持图像分类的训练与推理，适合多类别分类任务。

## 项目功能

- **图像数据预处理**：包括图像大小调整、随机翻转、旋转、裁剪、标准化等。
- **数据集划分**：将数据集分为训练集和验证集。
- **迁移学习**：加载预训练的 ResNet-50 模型，并微调最后的全连接层以适应新的分类任务。
- **训练与验证**：在训练集上训练模型，并在验证集上评估性能。
- **提前停止**：当验证损失不再减少时，自动停止训练，避免过拟合。
- **模型保存**：训练过程中的最佳模型会被保存，以便后续使用。
- **推理功能**：提供一个函数用于加载训练好的模型并对新的图像进行分类预测。
- **评估功能**：计算模型的准确率、精确度、召回率、F1分数，并绘制混淆矩阵、PR曲线和ROC曲线。

## 安装要求

确保已安装以下依赖项：

- Python 3.8+
- PyTorch 1.8.0+
- torchvision 0.9.0+
- PIL (Python Imaging Library)
- NumPy
- tqdm
- scikit-learn
- matplotlib
- seaborn

你可以通过 `requirements.txt` 安装所有依赖项：

```bash
pip install -r requirements.txt
```

## 数据集准备

请将数据集按以下结构组织：

```
data/
├── train/
│   ├── tomato/
│   │   ├── Tomato_Healthy/
│   │   ├── Tomato_Target_Spot/
│   │   ├── Tomato_Leaf_Mold/
├── test/
│   ├── tomato/
│   │   ├── Tomato_Healthy/
│   │   ├── Tomato_Target_Spot/
│   │   ├── Tomato_Leaf_Mold/
```

每个类别的图像应放在相应的文件夹中，`train` 文件夹用于训练集，`test` 文件夹用于测试集。

## 使用方法

### 1. 训练模型

运行以下命令开始训练模型：

```bash
python train.py
```

该命令将执行以下操作：

1. 加载训练数据和验证数据。
2. 使用预训练的 **ResNet-50** 模型进行迁移学习。
3. 显示训练损失、验证损失和准确率。
4. 最终保存最佳模型（`best_model.pth`）和最终模型（`final_model.pth`）。

### 2. 使用训练好的模型进行推理

运行以下代码使用训练好的模型进行图像分类预测：

```python
test_image_path = 'data/test/tomato/Tomato_Healthy/1.jpg'
predicted_class = predict_image(test_image_path)
print(f"Predicted class: {predicted_class}")
```

### 3. 评估模型

运行测试代码，计算模型的准确率、精确度、召回率、F1分数，并生成混淆矩阵、PR曲线和ROC曲线：

```bash
python test.py
```

该命令会在结果文件夹中生成相关评估图像，并打印评估结果。

## 目录结构

```
resnet-50/
├── data/                # 数据集文件夹
├── pth/                 # 保存训练好的模型
│   ├── tomato/
│   │   ├── best_model.pth  # 最佳模型
│   │   ├── final_model.pth # 最终模型
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── requirements.txt     # 项目依赖
└── README.md            # 项目说明
```

## 项目许可证

此项目使用 **GNU General Public License v3.0** 许可证。

## 致谢

感谢 [PyTorch](https://pytorch.org/) 和 [torchvision](https://pytorch.org/vision/stable/index.html) 提供的强大工具，使得深度学习模型的训练和推理变得更容易。