# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""
import os
import numpy as np

# 原始数据路径和输出数据路径
input_data_path = r'D:\Project-2\dataset\IXI-Aligned-DP130'
output_data_path = r'D:\Project-2\dataset\IXI-Aligned-DP130_80'

# 中间 80 个切片的索引范围
start_slice = 37
end_slice = 117  # 37到117共81个切片，实际截取是[37:117]，为80个切片

# 创建输出目录（如果不存在）
os.makedirs(output_data_path, exist_ok=True)

# 遍历 train、val 和 test 文件夹
for subset in ['train', 'val', 'test']:
    print(f"正在处理{subset}文件夹")
    subset_input_path = os.path.join(input_data_path, subset)
    subset_output_path = os.path.join(output_data_path, subset)
    os.makedirs(subset_output_path, exist_ok=True)

    for subject in os.listdir(subset_input_path):
        subject_input_path = os.path.join(subset_input_path, subject)
        subject_output_path = os.path.join(subset_output_path, subject)
        
        # 确保这是一个目录而不是文件
        if not os.path.isdir(subject_input_path):
            continue
            
        os.makedirs(subject_output_path, exist_ok=True)
        
        # 处理该受试者文件夹下的所有.npy文件
        for npy_file in os.listdir(subject_input_path):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(subject_input_path, npy_file)
                
                try:
                    # 加载.npy文件
                    data = np.load(file_path)
                    
                    # 检查维度是否足够提取所需切片
                    if data.shape[0] >= end_slice:
                        # 提取指定范围的切片
                        sliced_data = data[start_slice:end_slice]
                        # 复制到新位置，保持相同的文件名
                        np.save(os.path.join(subject_output_path, npy_file), sliced_data)
                        print(f"复制文件: {file_path} -> 形状从 {data.shape} 变为 {sliced_data.shape}")
                    else:
                        print(f"警告: 文件 {file_path} 的第一维为 {data.shape[0]}，小于所需的 {end_slice}，无法提取指定范围")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")

print("处理完成，已将每个.npy文件的切片[37:117]复制到新路径，保持目录结构与文件名不变。")


