import os
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 设置设备（如果有 GPU 可用，则使用 GPU，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义图像预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载测试集
test_dir = 'data/test/tomato'  # 指定测试集路径
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取类别名称
class_names = test_dataset.classes

# 加载已训练的模型
model = models.resnet50()
model.fc = nn.Linear(2048, len(class_names))  # 确保最后一层与训练时的设置相同
model.load_state_dict(torch.load('pth/tomato/final_model.pth'))  # 加载训练好的模型权重
model = model.to(device)
model.eval()  # 切换到评估模式

# 初始化存储标签和预测的列表
all_labels = []
all_predictions = []
all_probabilities = []

# 测试阶段
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        # 收集所有标签和预测值
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# 输出评估结果
print("--------------------------------------------------")
print("------------ Model Evaluation Results ------------")
print("--------------------------------------------------")
print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1 Score       : {f1 * 100:.2f}%")
print("--------------------------------------------------")


# 绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))  # 调整图形尺寸，确保足够的显示空间
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 12})

    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)

    # 调整布局防止标签被截断
    plt.xticks(rotation=45, ha='right', fontsize=12)  # 调整 x 轴标签角度和字体大小
    plt.yticks(rotation=0, fontsize=12)  # 调整 y 轴标签字体大小
    plt.tight_layout()

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # 使用 bbox_inches 确保保存时不会截断
        print(f"Confusion matrix saved at {save_path}")
    else:
        plt.show()
    plt.close()


# 绘制并保存 PR 曲线和 ROC 曲线
def plot_pr_roc_curves(y_true, y_scores, n_classes, class_names, save_dir=None):
    # 绘制 PR 曲线
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(label_binarize(y_true, classes=list(range(n_classes)))[:, i],
                                                      y_scores[:, i])
        plt.plot(recall, precision, label=f'Class {class_names[i]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid()

    if save_dir:
        pr_path = os.path.join(save_dir, "precision_recall_curve.png")
        plt.savefig(pr_path)
        print(f"Precision-Recall curve saved at {pr_path}")
    else:
        plt.show()
    plt.close()

    # 绘制 ROC 曲线
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(label_binarize(y_true, classes=list(range(n_classes)))[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.grid()

    if save_dir:
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path)
        print(f"ROC curve saved at {roc_path}")
    else:
        plt.show()
    plt.close()


# 创建保存目录
output_dir = 'results/tomato'
os.makedirs(output_dir, exist_ok=True)

# 绘制并保存混淆矩阵
plot_confusion_matrix(all_labels, all_predictions, class_names,
                      save_path=os.path.join(output_dir, "confusion_matrix.png"))

# 绘制并保存 PR 曲线和 ROC 曲线
plot_pr_roc_curves(all_labels, np.array(all_probabilities), len(class_names), class_names, save_dir=output_dir)
