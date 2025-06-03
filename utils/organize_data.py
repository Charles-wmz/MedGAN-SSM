import os
import shutil
from collections import defaultdict
from tqdm import tqdm

def organize_and_check_dataset(source_dir):
    # 获取所有文件
    files = os.listdir(source_dir)
    
    # 用字典存储每个受试者的模态文件
    subject_files = defaultdict(dict)
    
    # 解析文件名并按受试者分组
    print("正在分析文件...")
    for file in files:
        # 假设文件名格式为: IXI012-HH-1211-T1.nii.gz 或类似格式
        parts = file.split('-')
        if len(parts) >= 3:
            subject_id = parts[0]  # 例如 IXI012
            modality = parts[-1].split('.')[0].lower()  # 获取模态名称(t1/t2/pd)
            subject_files[subject_id][modality] = file

    # 创建受试者文件夹并移动文件
    print("\n开始组织文件...")
    missing_modalities = defaultdict(list)
    
    for subject_id, modalities in tqdm(subject_files.items(), desc="处理受试者"):
        # 创建受试者文件夹
        subject_dir = os.path.join(source_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # 检查并移动每个模态的文件
        for modality in ['t1', 't2', 'pd']:
            if modality in modalities:
                src = os.path.join(source_dir, modalities[modality])
                dst = os.path.join(subject_dir, modalities[modality])
                if not os.path.exists(dst):  # 避免重复移动
                    shutil.move(src, dst)
            else:
                missing_modalities[subject_id].append(modality)

    # 打印统计信息
    print("\n数据集组织完成！")
    print(f"总受试者数量: {len(subject_files)}")
    
    # 检查缺失模态
    subjects_with_missing = {k: v for k, v in missing_modalities.items() if v}
    if subjects_with_missing:
        print("\n发现以下受试者缺少模态:")
        for subject_id, missing in subjects_with_missing.items():
            print(f"{subject_id}: 缺少 {', '.join(missing)}")
        print(f"\n共有 {len(subjects_with_missing)} 个受试者存在模态缺失")
    else:
        print("\n所有受试者都具有完整的三个模态(T1, T2, PD)")

if __name__ == "__main__":
    source_dir = r"D:\Project-2\dataset\IXI"
    organize_and_check_dataset(source_dir)