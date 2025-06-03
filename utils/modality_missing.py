# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:54:21 2024

@author: D-03
"""

import os
import shutil
import numpy as np

# Remove a fixed modality for each subject in the given dataset split
def simulate_missing_modality_fixed(dataset_path, output_path, fixed_missing_modality):
    """
    Simulate missing modality by setting a fixed modality to zeros for each subject and saving to a new directory.
    
    Args:
        dataset_path (str): Path to the original dataset (train, val, or test).
        output_path (str): Path to save the modified dataset.
        fixed_missing_modality (str): Fixed modality to be set to zero (e.g., 't1.npy').
    """
    subjects = os.listdir(dataset_path)  # List all subjects in the split
    modalities = ['t1.npy', 't2.npy', 'pd.npy']  # IXI数据集的模态
    
    if fixed_missing_modality not in modalities:
        raise ValueError(f"Invalid modality: {fixed_missing_modality}. Must be one of {modalities}.")
    
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        output_subject_path = os.path.join(output_path, subject)
        
        if os.path.isdir(subject_path):
            # Copy the entire subject folder to the output path
            shutil.copytree(subject_path, output_subject_path, dirs_exist_ok=True)
            
            # Use the fixed missing modality
            missing_modality = fixed_missing_modality
            missing_modality_path = os.path.join(output_subject_path, missing_modality)
            
            if os.path.exists(missing_modality_path):
                # Load the modality data
                original_data = np.load(missing_modality_path)
                
                # Save original data in a dedicated folder within the output directory
                original_folder = os.path.join(output_subject_path, "original_modalities")
                if not os.path.exists(original_folder):
                    os.makedirs(original_folder)
                np.save(os.path.join(original_folder, f"original_{missing_modality}"), original_data)  # Save original data for evaluation
                
                # Set modality data to zero for training
                original_data.fill(0)
                np.save(missing_modality_path, original_data)
                
                # 判断是否为全零文件，并删除(将原本的数据设置为0，然后删除全为0的文件)
                if np.array_equal(original_data, np.zeros_like(original_data)):
                    os.remove(missing_modality_path)  # 删除全零的 `.npy` 文件
                    
            # Create or overwrite a metadata file to record which modality is missing
            metadata_file_path = os.path.join(output_subject_path, "missing_modality.txt")
            with open(metadata_file_path, "w") as f:
                f.write(f"{missing_modality}\n")

fixed_missing_modality = 't2.npy'  # 指定要缺失的模态 (e.g., 't1.npy', 't2.npy', 'pd.npy')
# Paths to the original preprocessed dataset
dataset_path = r"D:\Project-2\dataset\IXI-Aligned-DP130_80"  # IXI数据集路径
output_base_path = f"D:/Project-2/dataset/IXI-Aligned-DP130_80_SM{fixed_missing_modality}"  # 输出路径，根据缺失模态命名
# Simulate missing modality for each split (train, val, test)


for split in ['train', 'val', 'test']:
    print(f"正在处理{fixed_missing_modality}模态{split}文件夹")
    split_path = os.path.join(dataset_path, split)
    output_split_path = os.path.join(output_base_path, split)
    simulate_missing_modality_fixed(split_path, output_split_path, fixed_missing_modality)
    print(f"{fixed_missing_modality}模态模拟缺失已完成！")
