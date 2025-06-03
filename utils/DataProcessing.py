# -*- coding: utf-8 -*-
'''
用于预处理IXI数据集，进行N4偏置场校正、Z-score归一化、强度缩放到[-1,1]
'''

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import random
import shutil

# 数据集路径
data_path = r"D:\Project-2\dataset\IXI-Aligned"

# 输出路径（预处理数据的输出路径）
output_path = r"D:\Project-2\dataset\IXI-Aligned-DP130"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# N4 偏置场校正
def n4_bias_correction(input_image):
    sitk_image = sitk.ReadImage(input_image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)  # 将图像转换为 32 位浮点类型
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)

# Z-score 归一化
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# 强度缩放到 [-1, 1]
def intensity_scaling(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# 数据集分割函数
def split_dataset(dataset_path):
    """
    将数据集分割为训练集、验证集和测试集
    """
    # 创建分割后的数据集目录
    train_dir = os.path.join(os.path.dirname(dataset_path), "train")
    val_dir = os.path.join(os.path.dirname(dataset_path), "val")
    test_dir = os.path.join(os.path.dirname(dataset_path), "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 获取所有主体ID
    subject_ids = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    random.shuffle(subject_ids)
    
    # 划分比例: 70% 训练, 15% 验证, 15% 测试
    train_size = int(len(subject_ids) * 0.8)
    val_size = int(len(subject_ids) * 0.1)
    
    train_ids = subject_ids[:train_size]
    val_ids = subject_ids[train_size:train_size+val_size]
    test_ids = subject_ids[train_size+val_size:]
    
    # 分配数据
    for subject_id in train_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(train_dir, subject_id))
    
    for subject_id in val_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(val_dir, subject_id))
    
    for subject_id in test_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(test_dir, subject_id))

# IXI数据集预处理
def process_ixi_dataset():
    # 遍历IXI数据集下的所有文件
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # 获取模态类型（T1, T2, PD, MRA, DTI）
                modality = None
                subject_id = None
                
                # 解析文件名以获取模态和主体ID
                if 'T1' in file:
                    modality = 't1'
                elif 'T2' in file:
                    modality = 't2'
                elif 'PD' in file:
                    modality = 'pd'
                elif 'MRA' in file:
                    modality = 'mra'
                elif 'DTI' in file:
                    modality = 'dti'
                else:
                    continue  # 跳过不支持的模态
                
                # 从文件名中提取完整的主体ID (例如: IXI012-HH-1211)
                file_name_without_ext = file.split('.')[0]  # 移除扩展名
                subject_parts = file_name_without_ext.rsplit('-', 1)[0]  # 移除模态部分，保留完整ID
                subject_id = subject_parts  # 完整ID，如 IXI012-HH-1211
                
                # 完整的文件路径
                file_path = os.path.join(root, file)
                
                # 为主体创建输出文件夹，使用完整的主体ID
                subject_output_folder = os.path.join(output_path, subject_id)
                if not os.path.exists(subject_output_folder):
                    os.makedirs(subject_output_folder)
                
                # 如果该模态的处理结果已存在，则跳过
                output_file = os.path.join(subject_output_folder, f"{modality}.npy")
                if os.path.exists(output_file):
                    print(f"Skipping {file} as it is already processed.")
                    continue
                
                print(f"Processing {file}...")
                
                try:
                    # 应用预处理
                    n4_corrected = n4_bias_correction(file_path)
                    normalized = z_score_normalization(n4_corrected)
                    scaled = intensity_scaling(normalized)
                    
                    # 保存预处理后的数据
                    np.save(output_file, scaled)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

# 执行IXI数据集预处理
process_ixi_dataset()
print("Preprocessing complete.")

# 使用预处理后的输出路径进行数据集划分
split_dataset(output_path)
print("Splitting complete.")
