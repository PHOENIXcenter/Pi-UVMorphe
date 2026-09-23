import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import copy
import os
import time
from tqdm import tqdm
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import os
import re
from pathlib import Path
import json
import matplotlib.pyplot as plt
import io
import base64
from .mgf_parser import *


class MassSpecDataset(Dataset):
    def __init__(self, df, transform=None, tabular_columns=['Mass', 'Intensity', 'Charge', 'mz']):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tabular_columns = tabular_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tabular_features = row[self.tabular_columns].values.astype(np.float32)
        tabular_features = torch.from_numpy(tabular_features)
        image_path = row['image_path']
        return {
            'tabular': tabular_features,
            'image_path': image_path,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }


class CustomDataset(Dataset):
    def __init__(self, data_list, target_size=(224, 224), transform=None):
        self.data = data_list
        self.target_size = target_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tabular = item['tabular'].float()
        image_path = item['image_path']
        label = item['label'].long()

        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image_tensor = torch.zeros(4, self.target_size[0], self.target_size[1])

        return tabular, image_tensor, label


def load_images_for_training(envelope_dict, normalize=True):
    """
    加载所有图片为numpy数组，用于模型训练
    """
    images = []
    labels = []
    metadata = []

    for env_id, entry in envelope_dict.items():
        if 'image_array' in entry:
            img = entry['image_array']
            if normalize:
                img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(env_id)
            metadata.append(entry['info'])

    X = np.array(images)
    y = np.array(labels)

    print(f"Loaded {len(images)} images, shape: {X.shape}")
    print(f"Image value range: [{X.min():.3f}, {X.max():.3f}]")
    return X, y, metadata


def visualize_clean_image(envelope_dict, env_id):
    """显示干净的图片"""
    if env_id not in envelope_dict:
        print(f"Envelope {env_id} does not exist")
        return

    entry = envelope_dict[env_id]

    if 'image_array' in entry:
        img = entry['image_array']
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Envelope {env_id} (Clean for ViT)", fontsize=10)
        plt.show()
    elif 'image_path' in entry:
        img = plt.imread(entry['image_path'])
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Envelope {env_id} (Clean for ViT)", fontsize=10)
        plt.show()
    else:
        print("No available image data")


def create_dataset_for_vit(envelope_dict, output_dir='vit_dataset'):
    """
    创建可直接用于ViT训练的数据集文件夹结构
    """
    dataset_path = Path(output_dir)
    dataset_path.mkdir(exist_ok=True)

    # 为每个envelope创建子文件夹（按ID分类）
    for env_id, entry in envelope_dict.items():
        env_dir = dataset_path / str(env_id)
        env_dir.mkdir(exist_ok=True)

        if 'image_array' in entry:
            # 直接从数组保存
            img_path = env_dir / f'envelope_{env_id:03d}.png'
            from PIL import Image
            Image.fromarray(entry['image_array']).save(img_path)
        elif 'image_path' in entry:
            import shutil
            src = Path(entry['image_path'])
            dst = env_dir / src.name
            if not dst.exists():
                shutil.copy(src, dst)

    # 保存元数据JSON
    metadata = {}
    for env_id, entry in envelope_dict.items():
        metadata[env_id] = entry['info']

    with open(dataset_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {dataset_path}/")
    print(f"  - {len(envelope_dict)} subfolders")
    print(f"  - metadata.json contains metadata")


def get_torch_dataset(envelope_dict, transform=None):
    """
    返回可直接用于PyTorch的Dataset对象
    """
    from torch.utils.data import Dataset

    class SpectrumDataset(Dataset):
        def __init__(self, envelope_dict, transform=None):
            self.ids = list(envelope_dict.keys())
            self.envelope_dict = envelope_dict
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            env_id = self.ids[idx]
            entry = self.envelope_dict[env_id]

            # 获取图片
            if 'image_array' in entry:
                img = entry['image_array']
            else:
                from PIL import Image
                img = np.array(Image.open(entry['image_path']))

            # 转换为float32并归一化
            img = img.astype(np.float32) / 255.0

            # 转换为CHW格式 (C, H, W)
            img = np.transpose(img, (2, 0, 1))

            if self.transform:
                from PIL import Image
                img = Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0))
                img = self.transform(img)

            return img, env_id

    return SpectrumDataset(envelope_dict, transform)


def extract_file_prefix(file_name):
    """
    从文件名中提取前缀（去除_数字_ms2部分）
    例如: T186R_Dimer_17+_UVPDMS_1p_3mJ_1_ms2.msalign -> T186R_Dimer_17+_UVPDMS_1p_3mJ
    """
    # 匹配模式：去除 _数字_ms2 或 _ms2 后缀
    pattern = r'(_\d+_ms2|_ms2)\.msalign$'
    prefix = re.sub(pattern, '', file_name)
    return prefix


def group_files_by_prefix(directory, extension='.msalign'):
    """
    将目录下的所有.msalign文件按前缀分组
    返回字典：{前缀: [文件路径列表]}
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # 获取所有.msalign文件
    msalign_files = list(directory.glob(f'*{extension}'))

    if not msalign_files:
        print(f"No {extension} files found in {directory}")
        return {}

    # 按前缀分组
    groups = {}
    for file_path in msalign_files:
        prefix = extract_file_prefix(file_path.name)
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(str(file_path))

    return groups


def process_file_group(file_paths, prefix):
    """
    处理一组文件（相同前缀的多个副本）
    """
    print(f"\n{'=' * 60}")
    print(f"Processing file group: {prefix}")
    print(f"Contains {len(file_paths)} files:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")
    print(f"{'=' * 60}")

    # 初始化最终的列表
    spectra_dfs = []  # 存储所有文件的所有光谱
    mono_arr = []  # 存储所有文件的原始数据

    # 逐个处理文件
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file {i + 1}/{len(file_paths)}: {file_name}")

        # 注意：这里假设您已经定义了 extract_ms_info_from_mgf 函数
        # 根据您的文件格式，可能需要调整解析函数
        with open(file_path, 'r') as f:
            # 根据您的实际函数名调整
            file_mono_arr, file_spectra_dfs = extract_ms_info_from_mgf(f)

        # 添加到总列表中
        spectra_dfs.extend(file_spectra_dfs)
        mono_arr.extend(file_mono_arr)

        print(f"  ✓ Added {len(file_spectra_dfs)} spectra")

    print(f"\nFile group processing complete:")
    print(f"  Total spectra: {len(spectra_dfs)}")
    print(f"  Total raw data points: {len(mono_arr)}")

    # 执行交集查找
    result, updated_dfs = find_intersection(spectra_dfs)

    return result, updated_dfs, spectra_dfs, mono_arr


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for tabular, image, labels in tqdm(dataloader, desc='Training'):
        tabular = tabular.to(device)
        image = image.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 检查输入是否有nan
        if torch.isnan(tabular).any() or torch.isnan(image).any():
            print("Input data contains NaN values")
            continue

        if scaler:
            with autocast():
                outputs = model(tabular, image)
                loss = criterion(outputs, labels)

            # 检查loss
            # if torch.isnan(loss):
            # print(f"Loss is nan at batch")
            # continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 允许梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(tabular, image)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f"Loss is nan at batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


# 验证函数
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tabular, image, labels in tqdm(dataloader, desc='Validation'):
            tabular = tabular.to(device)
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(tabular, image)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, all_labels, all_preds
