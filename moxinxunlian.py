"""
基于JSON标注的西红柿病害检测深度学习模型
使用Mask R-CNN框架实现目标检测
"""

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# ====================== 配置参数 ======================
CONFIG = {
    "data_root": "./tomato_data",       # 数据集根目录
    "batch_size": 4,                    # 训练批次大小
    "num_epochs": 20,                   # 训练总轮次
    "learning_rate": 0.005,             # 初始学习率
    "image_size": (512, 512),           # 输入图像尺寸
    "num_classes": 6,                   # 类别数（5病害+背景）
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ====================== 数据解析模块 ======================
class LabelmeParser:
    """Labelme标注解析器"""
    def __init__(self):
        self.label_map = {
            'virus': 1,         # 病毒病
            'bacterial': 2,     # 青枯病
            'late_blight': 3,   # 晚疫病
            'gray_mold': 4,     # 灰霉病
            'early_blight': 5   # 早疫病
        }
        self.reverse_map = {v:k for k,v in self.label_map.items()}

    def parse_annotation(self, json_path):
        """解析单个标注文件"""
        with open(json_path) as f:
            data = json.load(f)
        
        # 获取图像路径
        img_path = Path(json_path).parent / data['imagePath']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB格式

        # 解析标注信息
        boxes = []
        labels = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                # 获取多边形顶点坐标
                points = np.array(shape['points'], dtype=np.float32)
                
                # 计算边界框 [x_min, y_min, x_max, y_max]
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                boxes.append([
                    x_coords.min(), 
                    y_coords.min(), 
                    x_coords.max(), 
                    y_coords.max()
                ])
                
                # 获取类别标签
                labels.append(self.label_map[shape['label']])
        
        return image, np.array(boxes), np.array(labels)

# ====================== 数据集类 ======================
class TomatoDataset(Dataset):
    """西红柿病害检测数据集"""
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.parser = LabelmeParser()
        self.transform = transform
        
        # 收集所有JSON标注文件
        self.json_files = list(self.root.glob('**/*.json'))
        
        # 数据过滤（可选）
        self.valid_files = []
        for json_path in self.json_files:
            try:
                image, boxes, labels = self.parser.parse_annotation(json_path)
                if len(boxes) > 0:
                    self.valid_files.append(json_path)
            except:  # noqa: E722
                print(f"无效标注文件：{json_path}")

    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        json_path = self.valid_files[idx]
        image, boxes, labels = self.parser.parse_annotation(json_path)
        
        # 数据增强
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # 转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        return image, target

# ====================== 数据增强 ======================
def get_transform(train=True):
    """获取数据增强管道"""
    if train:
        return A.Compose([
            A.Resize(*CONFIG['image_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(*CONFIG['image_size']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# ====================== 模型定义 ======================
class MaskRCNNModel(nn.Module):
    """改进的Mask R-CNN模型"""
    def __init__(self, num_classes):
        super().__init__()
        # 骨干网络（ResNet50 + FPN）
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        
        # 初始化Mask R-CNN
        self.model = MaskRCNN(
            backbone, 
            num_classes=num_classes,
            box_detections_per_img=200
        )
        
        # 添加特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1)
        )
    
    def forward(self, images, targets=None):
        # 特征提取
        features = self.model.backbone(images.tensors)
        
        # 特征增强
        enhanced_features = {
            k: self.feature_enhancer(v) 
            for k, v in features.items()
        }
        
        # 传入检测头
        return self.model(images, targets, enhanced_features)

# ====================== 训练流程 ======================
def train_model():
    # 初始化数据加载器
    train_dataset = TomatoDataset(
        CONFIG['data_root'], 
        transform=get_transform(train=True)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 初始化模型
    model = MaskRCNNModel(CONFIG['num_classes']).to(CONFIG['device'])
    
    # 优化器和学习率调度器
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG['learning_rate'],
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练循环
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        # 进度条显示
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]}')
        
        for images, targets in pbar:
            # 数据迁移到GPU
            images = [img.to(CONFIG['device']) for img in images]
            targets = [{k: v.to(CONFIG['device']) for k, v in t.items()} for t in targets]
            
            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # 更新统计信息
            epoch_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
        
        # 学习率调整
        lr_scheduler.step()
        
        # 打印统计信息
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} 平均损失: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'tomato_disease_model.pth')
    print('训练完成，模型已保存！')

# ====================== 推理模块 ======================
class DiseaseDetector:
    """病害检测器"""
    def __init__(self, model_path):
        self.model = MaskRCNNModel(CONFIG['num_classes']).to(CONFIG['device'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.parser = LabelmeParser()
        self.transform = get_transform(train=False)
    
    def predict(self, image_path, confidence=0.7):
        """执行预测"""
        # 读取图像
        if image_path.endswith('.json'):
            image, _, _ = self.parser.parse_annotation(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        transformed = self.transform(image=image, bboxes=[], labels=[])
        img_tensor = transformed['image'].unsqueeze(0).to(CONFIG['device'])
        
        # 推理
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        # 解析结果
        results = []
        for box, label, score in zip(
            predictions[0]['boxes'].cpu().numpy(),
            predictions[0]['labels'].cpu().numpy(),
            predictions[0]['scores'].cpu().numpy()
        ):
            if score > confidence:
                results.append({
                    'bbox': box.tolist(),
                    'disease': self.parser.reverse_map[label],
                    'confidence': float(score)
                })
        
        return results

# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 训练模型
    train_model()
    
    # 使用训练好的模型进行预测
    detector = DiseaseDetector('tomato_disease_model.pth')
    results = detector.predict('./test_image.jpg')
    
    # 打印结果
    print("检测结果：")
    for res in results:
        print(f"- {res['disease']}: 置信度 {res['confidence']:.2%}")