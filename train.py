import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np

# 设置设备（如果有 GPU 可用，则使用 GPU，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Step 1: 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Step 2: 加载数据
data_dir = 'data/train/tomato'
dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])

# 获取类别数
num_classes = len(dataset.classes)
print(f"Detected {num_classes} classes: {dataset.classes}")

# Step 3: 数据集划分为训练集和验证集
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
random_seed = 42  # 设置随机种子以确保可重复性
torch.manual_seed(random_seed)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 4: 加载预训练的 ResNet-50 模型
model = models.resnet50(pretrained=True)

# 冻结前几层，只训练 layer4 和 fc 层
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 修改最后的全连接层，调整输出类别数
model.fc = nn.Linear(2048, num_classes)
model = model.to(device)

# Step 5: 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Step 6: 训练模型
num_epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    torch.cuda.empty_cache()  # 清理 GPU 缓存

    # 训练阶段
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # 提前停止
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'pth/tomato/best_model.pth')
        print("Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    scheduler.step()

torch.save(model.state_dict(), 'pth/tomato/final_model.pth')
print("Final model saved.")


# Step 7: 推理函数
def predict_image(image_path):
    """对单张图像进行分类预测"""
    model.eval()
    image = Image.open(image_path)
    image = data_transforms['val'](image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return dataset.classes[predicted_class.item()]


# 示例测试：使用训练好的模型预测一张新图像的类别
test_image_path = 'data/test/tomato/Tomato_Healthy/4_aug_0.jpg'
predicted_class = predict_image(test_image_path)
print(f"Predicted class: {predicted_class}")
