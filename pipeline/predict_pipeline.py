import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from data.mgf_parser import extract_ms_info_from_mgf
from data.spectrum_parser import parse_spectrum_js
from data.image_generator import generate_envelope_dict_vit
from data.dataset import MassSpecDataset


class CustomDataset(Dataset):
    def __init__(self, data_list, target_size=(224, 224), transform=None):
        """
        Args:
            data_list: 包含字典的列表，每个字典包含 'tabular', 'image_path', 'label' 键
            target_size: 目标图像尺寸 (height, width)
            transform: 可选的图像变换（如果不提供，使用默认变换）
        """
        self.data = data_list
        self.target_size = target_size

        # 默认的图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.Resize(target_size),  # 调整大小
                transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tabular = item['tabular'].float()
        image_path = item['image_path']  # 图片路径
        label = item['label'].long()

        # 从路径加载图片
        try:
            # 使用PIL加载图片
            image = Image.open(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 应用变换（包括resize和转tensor）
            image_tensor = self.transform(image)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 如果加载失败，创建一个空白图片
            image_tensor = torch.zeros(4, self.target_size[0], self.target_size[1])

        return tabular, image_tensor, label


# ========== 预测新数据集 ==========
def predict(model, dataset, batch_size=64, device='cuda'):
    """
    对新数据集进行预测

    Args:
        model: 训练好的模型
        dataset: 要预测的数据集
        batch_size: 批次大小
        device: 设备

    Returns:
        predictions: 预测的类别 (numpy array)
        probabilities: 预测的概率 (numpy array)
    """
    model.eval()

    # 创建 DataLoader
    dataloader = DataLoader(
        CustomDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            # 假设你的 CustomDataset 返回 (tabular_data, image_data, labels)
            # 对于预测，可能没有标签，需要根据你的 CustomDataset 调整
            if len(batch) == 3:
                tabular_data, image_data, _ = batch
            else:
                tabular_data, image_data = batch

            tabular_data = tabular_data.to(device)
            image_data = image_data.to(device)

            # 前向传播
            outputs = model(tabular_data, image_data)

            # 获取预测类别和概率
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def load_model(model, model_path, device='cuda'):
    """加载训练好的模型权重"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded model from epoch {checkpoint['epoch']}, validation accuracy: {checkpoint['val_acc']:.4f}")
    return model


def process_single_file(
        msalign_file: str,
        spectrum_js_file: str,
        standard_scaler,
        model,
        device,
        output_envelope_dir: str = 'envelope_plots_temp',
        save_images: bool = True,
        return_df: bool = True
) -> pd.DataFrame:
    """处理单个质谱数据文件，返回预测结果的DataFrame"""

    if not os.path.exists(msalign_file):
        raise FileNotFoundError(f"The msalign file does not exist: {msalign_file}")
    if not os.path.exists(spectrum_js_file):
        raise FileNotFoundError(f"The spectrum0.js file does not exist: {spectrum_js_file}")

    os.makedirs(output_envelope_dir, exist_ok=True)

    # 解析 spectrum0.js
    print(f"Analysing spectrum0.js: {spectrum_js_file}")
    data = parse_spectrum_js(spectrum_js_file)
    envelopes = data.get('envelopes', [])
    print(f"Find {len(envelopes)} envelopes")

    # 生成 envelope 图片
    generate_envelope_dict_vit(
        data,
        output_dir=output_envelope_dir,
        img_size=(224, 224),
        dpi=100,
        save_images=save_images,
        save_base64=False,
        return_arrays=not save_images
    )
    print(f"✓ Image generation completed")

    # 读取 msalign
    with open(msalign_file, 'r') as f:
        mono_arr, spectra_dfs = extract_ms_info_from_mgf(f)
    print(f"Extract {len(spectra_dfs)} spectra")

    # 处理每个 spectrum
    all_results = []
    for idx, df in enumerate(spectra_dfs):
        print(f"\nprocessing spectrum {idx}")

        if save_images:
            image_files = sorted([
                f for f in os.listdir(output_envelope_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            image_files = [f"envelope_{i}.png" for i in range(len(df))]

        if len(image_files) == len(df) + 1:
            print(f"  ⚠️  Image count ({len(image_files)}) is one more than dataframe rows ({len(df)}). Removing the first image.")
            image_files = image_files[1:]  # 删除第一张图片
            print(f"  ✓ Image count after removal: {len(image_files)}")

        if len(image_files) != len(df):
            raise ValueError(f"Number of images ({len(image_files)}) != Number of rows in dataframe ({len(df)})")



        df_copy = df.copy()
        required_cols = ['Mass', 'Intensity', 'Charge', 'mz', 'label']
        for col in required_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0

        if save_images:
            image_paths = [os.path.join(output_envelope_dir, img) for img in image_files]
        else:
            image_paths = image_files



        df_copy['image_path'] = image_paths

        # 标准化
        feature_cols = ['Mass', 'Intensity', 'Charge', 'mz']
        scaled_df = df_copy.copy()
        scaled_df[feature_cols] = standard_scaler.transform(scaled_df[feature_cols])
        print("  ✓ Normalization completed")

        dataset = MassSpecDataset(df=scaled_df, tabular_columns=feature_cols)
        predictions, probabilities = predict(model, dataset, batch_size=64, device=device)

        scaled_df['pred_label'] = predictions
        if probabilities.ndim == 2:
            scaled_df['prob_0'] = probabilities[:, 0]
            scaled_df['prob_1'] = probabilities[:, 1]
        scaled_df['spectrum_idx'] = idx

        all_results.append(scaled_df)

    final_df = pd.concat(all_results, ignore_index=True)
    print(f"\n✓ Processing completed! {len(final_df)} records")

    return final_df