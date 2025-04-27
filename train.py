import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import os
from collections import Counter

# 定义自定义数据集类
class TomatoDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.transform = transform
        self.create_class_mapping(annotations)
        self.images = []
        self.labels = []
        self.load_images_and_labels(annotations)

    def create_class_mapping(self, annotations):
        categories = set()
        for annotation in annotations:
            label = annotation.get('label')
            if label:
                categories.add(label)
        
        self.classes = sorted(list(categories))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}

    def load_images_and_labels(self, annotations):
        for annotation in annotations:
            image_path = annotation.get('image_path')
            label = annotation.get('label')
            if image_path and label:
                label_idx = self.class_to_idx.get(label)
                if label_idx is not None:
                    self.images.append(image_path)
                    self.labels.append(label_idx)
                else:
                    print(f"Warning: Unknown label '{label}' in annotation: {annotation}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        print(f"Loading image: {image_path}")  # 调试信息
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 确保标签索引正确
        if label >= len(self.class_to_idx):
            raise ValueError(f"Label {label} is out of bounds. Class to index mapping: {self.class_to_idx}")

        return image, label

# 定义卷积神经网络模型
class TomatoDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(TomatoDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 定义早停类
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 定义训练函数
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping = EarlyStopping(patience=3, delta=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0    
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print()

            if phase == 'val':
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if early_stopping.early_stop:
            break

        if phase == 'train':
            scheduler.step()

    return model

# 准备数据
def prepare_data(json_files_dir, batch_size=1, test_size=0.2):
    annotations = []

    # 遍历不同病症的标注文件夹
    for disease_dir in os.listdir(json_files_dir):
        disease_path = os.path.join(json_files_dir, disease_dir)
        if os.path.isdir(disease_path):
            for file_name in os.listdir(disease_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(disease_path, file_name)
                    
                    # 加载单个标注文件
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # 提取图像路径和标签
                        image_path = data.get('imagePath')
                        label = data['shapes'][0]['label'] if data.get('shapes') else None
                        if image_path and label:
                            # 构造图像文件的绝对路径
                            # 假设图像文件存储在与标注文件夹同级的 images 文件夹中
                            image_path = os.path.join(json_files_dir, '..', 'images', disease_dir, image_path)
                            image_path = os.path.normpath(image_path)
                            
                            # 验证图像文件是否存在
                            if os.path.exists(image_path):
                                annotations.append({
                                    "image_path": image_path,
                                    "label": label
                                })
                            else:
                                print(f"Warning: Image file not found - {image_path}")
                        else:
                            print(f"Warning: Invalid annotation file - {file_path}")

    # 检查数据集大小
    if len(annotations) == 0:
        raise ValueError("No valid annotations found. Please check the JSON files and image paths.")

    # 检查类别分布
    labels = [annotation['label'] for annotation in annotations]
    label_counts = Counter(labels)
    print("Label counts:", label_counts)

    # 检查是否有类别只有一个样本
    for label, count in label_counts.items():
        if count < 2:
            print(f"Warning: Label '{label}' has only {count} samples.")

    # 如果数据集过小，调整划分比例
    if len(annotations) < 5:
        test_size = 0.1  # 减少验证集的比例
        print(f"Adjusting test_size to {test_size} due to small dataset size.")

    # 如果有类别只有一个样本，可以考虑不使用 stratify 参数
    if any(count < 2 for count in label_counts.values()):
        print("Warning: Some labels have only one sample. Skipping stratification.")
        train_annotations, val_annotations = train_test_split(
            annotations, 
            test_size=test_size, 
            random_state=42
        )
    else:
        train_annotations, val_annotations = train_test_split(
            annotations, 
            test_size=test_size, 
            random_state=42,
            stratify=labels
        )

    # 打印类别映射
    dataset = TomatoDataset(train_annotations)
    print("Class to index mapping:", dataset.class_to_idx)

    # 数据增强和预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset_train = TomatoDataset(train_annotations, transform=transform_train)
    dataset_val = TomatoDataset(val_annotations, transform=transform_val)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_val, dataset_train.class_to_idx

# 主函数
def main():
    # 数据路径（包含 JSON 标注文件的文件夹）
    json_files_dir = 'D:\\suanfa\\tomato-disease\\annotations'

    # 准备数据
    dataloader_train, dataloader_val, class_to_idx = prepare_data(json_files_dir)

    # 打印类别映射
    print("Class to index mapping:", class_to_idx)

    # 初始化模型
    num_classes = len(class_to_idx)
    model = TomatoDiseaseCNN(num_classes)

    # 打印模型结构
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25)

    # 保存模型权重和类别映射
    model_path = 'tomato_disease_model.pth'
    class_to_idx_path = 'class_to_idx.pkl'

    torch.save(model.state_dict(), model_path)
    with open(class_to_idx_path, 'wb') as f:
        pickle.dump(class_to_idx, f)

    print(f"Model saved to {model_path}")
    print(f"Class to index mapping saved to {class_to_idx_path}")

    # 模型评估
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader_val:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the validation images: {accuracy}%')

if __name__ == '__main__':
    main()