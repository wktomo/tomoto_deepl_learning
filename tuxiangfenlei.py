"""
优化版图像分类模型代码
改进点：
1. 增强数据预处理流程
2. 使用迁移学习框架
3. 优化训练策略
4. 增强模型推理功能
5. 改进代码健壮性
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import logging
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

# ====================== 配置参数 ======================
class Config:
    # 路径配置
    raw_data_path = "D:\\suanfa\\tomato-disease\\raw_images"        # 原始数据路径
    processed_path = "D:\\suanfa\\tomato-disease\\yuchuli"   # 预处理后路径
    model_save_dir = "D:\\suanfa\\tomato-disease\\moxingbaochun"           # 模型保存目录
    log_dir = "D:\\suanfa\\tomato-disease\\logs"                    # 日志目录
    
    # 模型参数
    target_size = (256, 256)             # 输入尺寸
    batch_size = 64                      # 批次大小
    num_epochs = 30                      # 训练轮数
    lr = 2e-4                            # 初始学习率
    num_classes = 5                      # 分类类别数
    early_stop_patience = 5              # 早停耐心值
    
    # 数据增强参数
    mixup_alpha = 0.2                    # MixUp参数
    
    # 类别映射（新增英文关键词映射）
    class_info = {
        "viral disease": {"id": 0, "name": "病毒病"},
        "bacterial wilt": {"id": 1, "name": "青枯病"},
        "late blight": {"id": 2, "name": "晚疫病"},
        "gray mold": {"id": 3, "name": "灰霉病"},
        "early blight": {"id": 4, "name": "早疫病"}
    }

# ====================== 初始化设置 ======================
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.log_dir, f"training_{datetime.now().strftime('%Y%m%d%H%M')}.log")),
        logging.StreamHandler()
    ]
)

# 自动创建目录
os.makedirs(Config.model_save_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)

# ====================== 增强数据预处理 ======================
class AdvancedDataPreprocessor:
    @staticmethod
    def organize_dataset():
        """改进的数据集整理方法"""
        try:
            logging.info("开始整理数据集...")
            
            # 创建目录结构
            for split in ['train', 'val']:
                for cls_info in Config.class_info.values():
                    dir_path = os.path.join(Config.processed_path, split, str(cls_info["id"]))
                    os.makedirs(dir_path, exist_ok=True)

            # 收集并分类文件
            all_files = []
            for root, _, files in os.walk(Config.raw_data_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_files.append(os.path.join(root, file))

            # 改进的文件分类逻辑
            file_dict = {cls_info["id"]: [] for cls_info in Config.class_info.values()}
            unlabeled_files = []
            
            for file_path in all_files:
                filename = os.path.basename(file_path).lower()
                matched = False
                for keyword, cls_info in Config.class_info.items():
                    if keyword in filename:
                        file_dict[cls_info["id"]].append(file_path)
                        matched = True
                        break
                if not matched:
                    unlabeled_files.append(filename)

            if unlabeled_files:
                logging.warning(f"发现{len(unlabeled_files)}个未分类文件，示例：{unlabeled_files[:3]}")

            # 划分数据集并复制文件
            for cls_id, files in file_dict.items():
                if len(files) == 0:
                    logging.warning(f"类别{cls_id}没有找到任何文件，请检查数据！")
                    continue
                
                train_files, val_files = train_test_split(
                    files, 
                    test_size=0.2,
                    random_state=42,
                    stratify=[cls_id]*len(files)  # 保持类别分布
                )
                
                AdvancedDataPreprocessor._copy_files(train_files, 'train', cls_id)
                AdvancedDataPreprocessor._copy_files(val_files, 'val', cls_id)

            logging.info("数据集整理完成！")

        except Exception as e:
            logging.error(f"数据集整理失败：{str(e)}")
            raise

    @staticmethod
    def _copy_files(files, split, cls_id):
        """安全的文件复制方法"""
        dest_dir = os.path.join(Config.processed_path, split, str(cls_id))
        for src in files:
            try:
                # 验证图像完整性
                with Image.open(src) as img:
                    img.verify()
                shutil.copy(src, dest_dir)
            except (IOError, SyntaxError, UnidentifiedImageError) as e:
                logging.warning(f"损坏文件跳过：{src} - {str(e)}")
            except Exception as e:
                logging.error(f"复制文件失败：{src} - {str(e)}")

    @classmethod
    def get_transforms(cls):
        """动态数据增强配置"""
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: ImageOps.exif_transpose(img)),
            transforms.RandomResizedCrop(Config.target_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
        ])

        val_transform = transforms.Compose([
            transforms.Resize(Config.target_size),
            transforms.CenterCrop(Config.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, val_transform

# ====================== 改进模型结构 ======================
class AdvancedCNN(nn.Module):
    """改进的CNN架构，结合预训练模型"""
    def __init__(self):
        super().__init__()
        # 使用预训练的EfficientNet作为骨干网络
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # 冻结底层参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 替换分类头
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, Config.num_classes)
        )
        
        # 辅助分类器
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1536, 256),  # 根据实际特征图尺寸调整
            nn.ReLU(),
            nn.Linear(256, Config.num_classes)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        main_output = self.backbone.classifier(features.flatten(start_dim=1))
        
        # 辅助输出
        aux_output = self.aux_classifier(features)
        return main_output + 0.3*aux_output

# ====================== 增强训练模块 ======================
class AdvancedTrainer:
    def __init__(self):
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备：{self.device}")
        
        # 数据加载
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        # 混合精度训练
        self.scaler = amp.GradScaler(enabled=self.device.type == "cuda")
        
        # 模型初始化
        self.model = AdvancedCNN().to(self.device)
        self._init_optimizer()
        
        # 训练状态跟踪
        self.best_val_acc = 0.0
        self.no_improve_epochs = 0

    def _init_dataloaders(self):
        """初始化数据加载器"""
        train_transform, val_transform = AdvancedDataPreprocessor.get_transforms()
        
        try:
            # 使用ImageFolder数据集
            train_data = datasets.ImageFolder(
                os.path.join(Config.processed_path, 'train'),
                transform=train_transform
            )
            val_data = datasets.ImageFolder(
                os.path.join(Config.processed_path, 'val'),
                transform=val_transform
            )
            
            # 数据加载器配置
            train_loader = DataLoader(
                train_data, 
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_data, 
                batch_size=Config.batch_size,
                num_workers=2,
                pin_memory=True
            )
            return train_loader, val_loader
        except Exception as e:
            logging.error(f"数据加载失败：{str(e)}")
            raise

    def _init_optimizer(self):
        """初始化优化器和调度器"""
        params = [
            {"params": self.model.backbone.classifier.parameters(), "lr": Config.lr},
            {"params": self.model.aux_classifier.parameters(), "lr": Config.lr*2}
        ]
        self.optimizer = optim.AdamW(params, weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode="max", 
            factor=0.5, 
            patience=2,
            verbose=True
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def _mixup_data(self, x, y, alpha=1.0):
        """MixUp数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def _train_epoch(self, epoch):
        """改进的训练epoch逻辑"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        processed = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # MixUp增强
            inputs, targets_a, targets_b, lam = self._mixup_data(inputs, targets, Config.mixup_alpha)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast(enabled=self.device.type == "cuda"):
                outputs = self.model(inputs)
                loss = lam * self.criterion(outputs, targets_a) + \
                      (1 - lam) * self.criterion(outputs, targets_b)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计信息
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (lam * preds.eq(targets_a).sum().item() + 
                       (1 - lam) * preds.eq(targets_b).sum().item())
            processed += inputs.size(0)
            
            progress_bar.set_postfix({
                "Loss": f"{total_loss/processed:.4f}",
                "Acc": f"{correct/processed:.4f}"
            })
        
        return total_loss / len(self.train_loader.dataset), correct / len(self.train_loader.dataset)

    def _validate(self):
        """改进的验证逻辑"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
        
        avg_loss = val_loss / len(self.val_loader.dataset)
        accuracy = correct / len(self.val_loader.dataset)
        return avg_loss, accuracy

    def train(self):
        """完整的训练流程"""
        logging.info("开始模型训练...")
        self.best_val_acc = 0.0
        
        for epoch in range(Config.num_epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 验证阶段
            val_loss, val_acc = self._validate()
            
            # 学习率调整
            self.scheduler.step(val_acc)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.no_improve_epochs = 0
                model_path = os.path.join(
                    Config.model_save_dir,
                    f"best_model_acc{val_acc:.4f}.pth"
                )
                torch.save(self.model.state_dict(), model_path)
                logging.info(f"保存新的最佳模型，准确率：{val_acc:.4f}")
            else:
                self.no_improve_epochs += 1
            
            # 早停判断
            if self.no_improve_epochs >= Config.early_stop_patience:
                logging.info(f"早停触发，连续{Config.early_stop_patience}轮未提升")
                break
            
            # 打印日志
            logging.info(
                f"Epoch {epoch+1}/{Config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
        
        logging.info(f"训练完成，最佳验证准确率：{self.best_val_acc:.4f}")

# ====================== 增强推理模块 ======================
class AdvancedPredictor:
    def __init__(self, model_path=None):
        # 自动加载最新模型
        if model_path is None:
            model_list = sorted(
                [f for f in os.listdir(Config.model_save_dir) if f.endswith(".pth")],
                key=lambda x: float(x.split("acc")[1].replace(".pth","")),
                reverse=True
            )
            if model_list:
                model_path = os.path.join(Config.model_save_dir, model_list[0])
            else:
                raise FileNotFoundError("没有找到可用的模型文件")
        
        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdvancedCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 预处理流程
        self.transform = transforms.Compose([
            transforms.Resize(Config.target_size),
            transforms.CenterCrop(Config.target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 类别映射
        self.id_to_name = {
            cls_info["id"]: cls_info["name"]
            for cls_info in Config.class_info.values()
        }

    def predict(self, input_path, confidence_threshold=0.7):
        """改进的预测方法，支持多种输入类型"""
        if os.path.isdir(input_path):
            return self._predict_batch(input_path, confidence_threshold)
        else:
            return self._predict_single(input_path, confidence_threshold)

    def _predict_single(self, image_path, confidence_threshold):
        """单张图像预测"""
        try:
            # 加载并预处理图像
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                input_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # 推理
                with torch.no_grad(), amp.autocast():
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                
                # 解析结果
                max_prob, pred_idx = torch.max(probs, dim=1)
                if max_prob.item() < confidence_threshold:
                    return "不确定"
                
                return self.id_to_name[pred_idx.item()]
        except Exception as e:
            logging.error(f"预测失败：{image_path} - {str(e)}")
            return "预测错误"

    def _predict_batch(self, folder_path, confidence_threshold):
        """批量预测"""
        results = {}
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(folder_path, filename)
                results[filename] = self._predict_single(file_path, confidence_threshold)
        
        return results

    def visualize_prediction(self, image_path, save_path=None):
        """可视化预测结果"""
        with Image.open(image_path) as img:
            plt.figure(figsize=(10, 6))
            
            # 显示原图
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("原图")
            plt.axis("off")
            
            # 显示预处理后的图像
            plt.subplot(1, 2, 2)
            processed_img = self.transform(img).cpu().permute(1, 2, 0).numpy()
            processed_img = processed_img * np.array([0.229, 0.224, 0.225]) + \
                            np.array([0.485, 0.456, 0.406])  # 反归一化
            plt.imshow(np.clip(processed_img, 0, 1))
            plt.title("预处理视图")
            plt.axis("off")
            
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
            else:
                plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    try:
        # 数据预处理
        AdvancedDataPreprocessor.organize_dataset()
        
        # 初始化训练器
        trainer = AdvancedTrainer()
        
        # 开始训练
        trainer.train()
        
        # 示例推理
        predictor = AdvancedPredictor()
        test_image = "./test_image.jpg"
        result = predictor.predict(test_image)
        print(f"预测结果：{result}")
        
        # 可视化示例
        predictor.visualize_prediction(test_image)
        
    except Exception as e:
        logging.error(f"程序运行异常：{str(e)}")
        raise