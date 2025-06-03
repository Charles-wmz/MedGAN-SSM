'''
用于配准IXI数据集中的T1、T2和PD模态，以T2为基准，将T1和PD对齐到T2
'''

import os
import SimpleITK as sitk
import argparse
import time
import re

def register_images(fixed_image_path, moving_image_path, output_path):
    """
    使用仿射变换对两个MRI图像进行配准
    
    参数:
        fixed_image_path: 参考图像路径
        moving_image_path: 需要对齐的图像路径
        output_path: 对齐后图像的保存路径
    """
    # 读取图像
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # 初始化配准框架
    registration_method = sitk.ImageRegistrationMethod()
    
    # 设置相似度度量方式 - 互信息(适合多模态)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    
    # 设置插值方法
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # 设置优化器
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                     numberOfIterations=100,
                                                     convergenceMinimumValue=1e-6, 
                                                     convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # 设置变换类型为仿射变换
    transform = sitk.AffineTransform(fixed_image.GetDimension())
    registration_method.SetInitialTransform(transform)
    
    # 执行配准
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # 应用变换
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    aligned_image = resampler.Execute(moving_image)
    
    # 保存结果
    sitk.WriteImage(aligned_image, output_path)
    
    return final_transform

def get_patient_id(filename):
    """从文件名中提取患者ID，例如从'IXI027-Guys-0710-T1.nii'提取'IXI027-Guys-0710'"""
    # 使用正则表达式匹配患者ID部分
    match = re.match(r'(IXI\d+-\w+-\d+)-[TP][D12]\.nii(?:\.gz)?', filename)
    if match:
        return match.group(1)
    return None

def process_ixi_dataset(dataset_dir, output_dir, limit=None):
    """
    处理IXI数据集中的所有图像，将T1和PD模态对齐到T2模态
    
    参数:
        dataset_dir: IXI数据集根目录，包含所有受试者文件夹
        output_dir: 对齐后图像的输出目录
        limit: 限制处理的受试者数量(用于测试)
    """
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有受试者文件夹
    subject_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    subject_dirs.sort()  # 确保处理顺序一致
    
    if limit and limit > 0:
        subject_dirs = subject_dirs[:limit]
    
    total_subjects = len(subject_dirs)
    print(f"找到 {total_subjects} 个受试者文件夹，开始进行对齐处理...")
    
    start_time = time.time()
    processed_count = 0
    success_count = 0
    
    for subject_dir in subject_dirs:
        subject_path = os.path.join(dataset_dir, subject_dir)
        
        # 在受试者目录下查找T1、T2和PD文件
        files = os.listdir(subject_path)
        t1_file = next((f for f in files if 'T1' in f and f.endswith('.nii.gz')), None)
        t2_file = next((f for f in files if 'T2' in f and f.endswith('.nii.gz')), None)
        pd_file = next((f for f in files if 'PD' in f and f.endswith('.nii.gz')), None)
        
        if not t2_file:
            print(f"警告: 在目录 {subject_dir} 中未找到T2文件，跳过该受试者")
            continue
            
        # 创建该受试者的输出目录
        subject_output_dir = os.path.join(output_dir, subject_dir)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # 构建完整的文件路径
        t2_path = os.path.join(subject_path, t2_file)
        t2_output = os.path.join(subject_output_dir, t2_file)
        
        # 复制T2图像到输出目录
        try:
            sitk.WriteImage(sitk.ReadImage(t2_path), t2_output)
        except Exception as e:
            print(f"复制 {subject_dir} 的T2图像时出错: {e}")
            continue
        
        # 对齐T1到T2
        if t1_file:
            t1_path = os.path.join(subject_path, t1_file)
            t1_output = os.path.join(subject_output_dir, t1_file)
            
            try:
                print(f"[{processed_count+1}/{total_subjects}] 正在对齐 {subject_dir} 的T1图像...")
                register_images(t2_path, t1_path, t1_output)
                success_count += 1
            except Exception as e:
                print(f"对齐 {subject_dir} 的T1图像时出错: {e}")
        else:
            print(f"警告: 未找到受试者 {subject_dir} 的T1图像")
            
        # 对齐PD到T2
        if pd_file:
            pd_path = os.path.join(subject_path, pd_file)
            pd_output = os.path.join(subject_output_dir, pd_file)
            
            try:
                print(f"[{processed_count+1}/{total_subjects}] 正在对齐 {subject_dir} 的PD图像...")
                register_images(t2_path, pd_path, pd_output)
                success_count += 1
            except Exception as e:
                print(f"对齐 {subject_dir} 的PD图像时出错: {e}")
        else:
            print(f"警告: 未找到受试者 {subject_dir} 的PD图像")
            
        processed_count += 1
        
        # 计算并显示进度
        if processed_count % 5 == 0 or processed_count == total_subjects:
            elapsed_time = time.time() - start_time
            avg_time_per_subject = elapsed_time / processed_count
            remaining_subjects = total_subjects - processed_count
            estimated_time = remaining_subjects * avg_time_per_subject
            
            print(f"进度: {processed_count}/{total_subjects} ({processed_count/total_subjects*100:.1f}%)")
            print(f"估计剩余时间: {estimated_time/60:.1f} 分钟")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n对齐处理完成!")
    print(f"总处理时间: {total_time/60:.1f} 分钟")
    print(f"成功对齐图像: {success_count}/{total_subjects*2} (T1+PD)")

def main():
    parser = argparse.ArgumentParser(description='对齐IXI数据集中的多模态MRI图像')
    
    # 设置默认路径
    default_dataset_path = r"D:\Project-2\dataset\IXI"
    default_output_dir = os.path.join(os.path.dirname(default_dataset_path), "IXI-Aligned")
    
    parser.add_argument('--dataset_dir', default=default_dataset_path, help='IXI数据集根目录')
    parser.add_argument('--output_dir', default=default_output_dir, help='输出目录')
    parser.add_argument('--limit', type=int, help='限制处理的受试者数量(用于测试)')
    
    args = parser.parse_args()
    
    print("MRI模态对齐工具 - IXI数据集")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.limit:
        print(f"限制处理数量: {args.limit}个受试者")
    
    # 确保数据集目录存在
    if not os.path.exists(args.dataset_dir):
        print(f"错误: 目录 {args.dataset_dir} 不存在!")
        return
    
    process_ixi_dataset(args.dataset_dir, args.output_dir, args.limit)

if __name__ == "__main__":
    main() 