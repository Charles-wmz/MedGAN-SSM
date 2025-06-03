import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，不需要图形界面
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
from datetime import datetime
# 导入Generator类，避免重复定义
from mainv2 import Generator

# 复制修改后的BRATSDataset类，添加subject_id属性
class VisualBRATSDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, phase='test', input_modalities=None, force_target_modality=None):
        self.data_path = os.path.join(base_path, phase)
        self.subjects = [d for d in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, d))]
        
        # 用户指定的输入模态和目标模态
        self.input_modalities = input_modalities  # 如果为None，则使用所有非缺失模态
        self.force_target_modality = force_target_modality  # 如果为None，则使用missing_modality.txt中的缺失模态
        
        # 所有可能的模态名称
        self.possible_modalities = ['t1', 't1ce', 't2', 'flair', 'pd']
        
        # 检测数据集类型
        dataset_name = os.path.basename(base_path).lower()
        self.is_brats = 'brats' in dataset_name
        self.is_ixi = 'ixi' in dataset_name
        
        # 根据数据集类型确定有效模态
        if self.is_brats:
            self.all_modalities = ['t1', 't1ce', 't2', 'flair']
        elif self.is_ixi:
            self.all_modalities = ['t1', 't2', 'pd']
        else:
            # 默认情况，使用所有可能的模态
            self.all_modalities = self.possible_modalities.copy()
        
        if self.input_modalities:
            for mod in self.input_modalities:
                if mod not in self.possible_modalities:
                    raise ValueError(f"无效的输入模态: {mod}。有效的模态为: {self.possible_modalities}")
        
        if self.force_target_modality and self.force_target_modality not in self.possible_modalities:
            raise ValueError(f"无效的目标模态: {self.force_target_modality}。有效的模态为: {self.possible_modalities}")
            
        # 检查输入模态和目标模态是否有冲突
        if self.input_modalities and self.force_target_modality:
            if self.force_target_modality in self.input_modalities:
                raise ValueError(f"目标模态 {self.force_target_modality} 不能同时作为输入模态")
        
    def __len__(self):
        return len(self.subjects) * 80  # 假设每个体积有80个切片
    
    def __getitem__(self, idx):
        # 计算实际的subject索引和切片索引
        # 假设每个体积有80个切片
        slice_per_volume = 80
        subj_idx = idx // slice_per_volume
        slice_idx = idx % slice_per_volume
        
        subj = self.subjects[subj_idx]
        # 加载模态数据
        modalities = {}
        
        # 读取缺失模态信息
        with open(os.path.join(self.data_path, subj, 'missing_modality.txt')) as f:
            missing_from_file = f.read().splitlines()
        
        # 确定实际缺失的模态
        if self.force_target_modality:
            # 如果用户指定了目标模态，则该模态被视为"缺失"（需要预测）
            missing_modality = self.force_target_modality
        else:
            # 否则使用文件中指定的缺失模态
            missing_modality = missing_from_file[0] if missing_from_file else None
        
        # 加载所有可用的模态数据
        for mod in self.all_modalities:
            if mod != missing_modality:  # 不加载目标模态（缺失模态）
                try:
                    path = os.path.join(self.data_path, subj, f'{mod}.npy')
                    volume = np.load(path)
                    # 提取特定切片
                    modalities[mod] = volume[slice_idx]
                except FileNotFoundError:
                    # 只有当模态应该存在于当前数据集中时才打印警告
                    if ((self.is_brats and mod in ['t1', 't1ce', 't2', 'flair']) or 
                        (self.is_ixi and mod in ['t1', 't2', 'pd'])):
                        print(f"警告: 无法找到模态 {mod} 的数据文件: {path}")
                    # 否则静默忽略
        
        # 确定输入模态
        if self.input_modalities:
            # 用户指定了输入模态
            input_mods = self.input_modalities
            # 检查指定的输入模态是否可用
            for mod in input_mods:
                if mod not in modalities:
                    # 如果是不应该存在于当前数据集的模态，提供更明确的错误信息
                    if ((self.is_brats and mod not in ['t1', 't1ce', 't2', 'flair']) or 
                        (self.is_ixi and mod not in ['t1', 't2', 'pd'])):
                        raise RuntimeError(f"指定的输入模态 {mod} 不适用于当前数据集类型。BraTS支持: t1, t1ce, t2, flair。IXI支持: t1, t2, pd。")
                    else:
                        raise RuntimeError(f"指定的输入模态 {mod} 不可用。可能是文件缺失或该模态是目标模态。")
        else:
            # 使用所有非目标模态作为输入
            input_mods = [mod for mod in self.all_modalities if mod != missing_modality and mod in modalities]
        
        # 确保至少有一个输入模态
        if not input_mods:
            raise RuntimeError(f"没有可用的输入模态。请检查数据或修改输入模态设置。")
        
        # 构建输入数据 - 注意现在是2D切片的堆叠
        try:
            input_data = np.stack([modalities[mod] for mod in input_mods], axis=0)
        except KeyError as e:
            raise RuntimeError(f"构建输入数据时出错: 模态 {e} 不可用")
        
        # 加载目标数据
        if missing_modality:
            try:
                target_path = os.path.join(self.data_path, subj, 'original_modalities', f'original_{missing_modality}.npy')
                volume = np.load(target_path)
                # 提取特定切片
                target = volume[slice_idx]
            except FileNotFoundError:
                raise RuntimeError(f"无法找到目标模态 {missing_modality} 的原始数据: {target_path}")
        else:
            # 如果没有缺失模态，默认使用flair作为目标
            missing_modality = 'flair'
            try:
                target_path = os.path.join(self.data_path, subj, 'original_modalities', f'original_{missing_modality}.npy')
                volume = np.load(target_path)
                # 提取特定切片
                target = volume[slice_idx]
            except FileNotFoundError:
                raise RuntimeError(f"无法找到默认目标模态 {missing_modality} 的原始数据: {target_path}")
        
        # 额外返回受试者ID和切片索引
        return {
            'input': torch.FloatTensor(input_data),
            'target': torch.FloatTensor(target).unsqueeze(0),  # 添加通道维度
            'missing_modality': missing_modality,
            'input_modalities': input_mods,
            'subject_id': subj,         # 返回原始受试者ID
            'slice_idx': slice_idx      # 返回切片索引
        }

def visualize_results(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 添加一个辅助函数，将NumPy类型转换为Python标准类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    # 初始化测试数据集，使用args中的参数
    test_dataset = VisualBRATSDataset(args.data_path, 'test', 
                                     args.input_modalities, 
                                     args.target_modality)
    
    # 确定输入通道数 - 从第一个样本获取
    sample = test_dataset[0]
    in_channels = sample['input'].shape[0]
    
    # 初始化模型
    model = Generator(in_channels=in_channels).to(device)
    
    # 加载模型权重
    if args.checkpoint_file is None:
        # 使用最新的checkpoint文件
        checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir) 
                                 if f.endswith('.pth') and os.path.isfile(os.path.join(args.checkpoint_dir, f))])
        if not checkpoint_files:
            raise ValueError(f"错误: 在 {args.checkpoint_dir} 中未找到checkpoint文件")
        
        checkpoint_file = checkpoint_files[-1]
    else:
        checkpoint_file = args.checkpoint_file
    
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
    print(f"加载模型权重: {checkpoint_path}")
    
    # 加载权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['G_state_dict'])
        print("成功加载模型权重")
        
        # 获取epoch信息，如果不存在则使用'unknown'
        epoch = checkpoint.get('epoch', 'unknown')
        
        # 打印checkpoint相关信息
        print(f"模型训练轮数: {epoch}")
        if 'psnr' in checkpoint:
            print(f"模型验证集PSNR: {checkpoint['psnr']:.2f}")
        if 'ssim' in checkpoint:
            print(f"模型验证集SSIM: {checkpoint['ssim']:.4f}")
        if 'mae' in checkpoint:
            print(f"模型验证集MAE: {checkpoint['mae']:.4f}")
        if 'input_modalities' in checkpoint and 'target_modality' in checkpoint:
            print(f"模型输入模态: {checkpoint['input_modalities']}")
            print(f"模型目标模态: {checkpoint['target_modality']}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 创建可视化结果保存目录 - 使用epoch而不是timestamp
    vis_dir = os.path.join(args.checkpoint_dir, f"visualization_epoch{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"可视化结果将保存在: {vis_dir}")
    
    # 用于存储评估指标
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }
    
    # 存储受试者级别的指标
    subject_metrics = {}
    
    # 设置为评估模式
    model.eval()
    
    # 逐个处理测试数据
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataset, desc="生成可视化结果")):
            # 获取受试者ID和切片索引
            subject_id = data['subject_id']
            slice_idx = data['slice_idx']
            
            # 创建该受试者的目录（如果尚未创建）
            subject_dir = os.path.join(vis_dir, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            # 初始化该受试者的指标（如果尚未初始化）
            if subject_id not in subject_metrics:
                subject_metrics[subject_id] = {
                    'psnr': [],
                    'ssim': [],
                    'mae': []
                }
            
            # 获取输入和目标数据
            input_data = data['input'].unsqueeze(0).to(device)  # 添加batch维度
            target_data = data['target'].unsqueeze(0).to(device)  # 添加batch维度
            
            # 模型推断
            output_data = model(input_data)
            
            # 计算评估指标
            target_np = target_data.cpu().numpy()[0]  # 移除batch维度
            output_np = output_data.cpu().numpy()[0]  # 移除batch维度
            
            # 计算PSNR
            psnr_value = peak_signal_noise_ratio(target_np, output_np)
            metrics['psnr'].append(psnr_value)
            subject_metrics[subject_id]['psnr'].append(psnr_value)
            
            # 计算SSIM
            try:
                ssim_value = structural_similarity(
                    target_np[0], output_np[0],
                    data_range=1.0,
                    win_size=min(7, min(target_np.shape[1], target_np.shape[2]) - 1),
                    multichannel=False
                )
                metrics['ssim'].append(ssim_value)
                subject_metrics[subject_id]['ssim'].append(ssim_value)
            except Exception as e:
                print(f"SSIM计算错误: {e}")
                metrics['ssim'].append(0.0)
                subject_metrics[subject_id]['ssim'].append(0.0)
            
            # 计算MAE
            mae_value = np.abs(target_np - output_np).mean()
            metrics['mae'].append(mae_value)
            subject_metrics[subject_id]['mae'].append(mae_value)
            
            # 保存输出图像和真实图像
            output_img = output_data[0, 0].cpu().numpy()
            target_img = target_data[0, 0].cpu().numpy()
            
            # 检查图像维度
            if output_img.size == 0 or target_img.size == 0:
                print(f"警告: 跳过空图像 - 受试者 {subject_id}, 切片 {slice_idx}")
                continue
                
            if output_img.shape[0] == 0 or output_img.shape[1] == 0 or \
               target_img.shape[0] == 0 or target_img.shape[1] == 0:
                print(f"警告: 跳过无效维度图像 - 受试者 {subject_id}, 切片 {slice_idx}")
                print(f"输出图像维度: {output_img.shape}, 目标图像维度: {target_img.shape}")
                continue
            
            # 保存输出图像
            plt.figure()
            plt.imshow(output_img, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(subject_dir, f'slice_{slice_idx:04d}_output.png'), 
                       bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 保存真实图像
            plt.figure()
            plt.imshow(target_img, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(subject_dir, f'slice_{slice_idx:04d}_target.png'), 
                       bbox_inches='tight', pad_inches=0)
            plt.close()
    
    # 计算平均指标
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # 计算每个受试者的平均指标
    subject_avg_metrics = {}
    for subj, subj_metrics in subject_metrics.items():
        subject_avg_metrics[subj] = {k: np.mean(v) for k, v in subj_metrics.items()}
    
    # 计算所有受试者之间的均值和方差
    subject_stats = {}
    # 收集所有受试者的平均PSNR、SSIM和MAE值
    psnr_across_subjects = [subj_avg['psnr'] for subj_avg in subject_avg_metrics.values()]
    ssim_across_subjects = [subj_avg['ssim'] for subj_avg in subject_avg_metrics.values()]
    mae_across_subjects = [subj_avg['mae'] for subj_avg in subject_avg_metrics.values()]
    
    # 计算受试者间的均值和方差
    subject_stats = {
        'psnr_mean': np.mean(psnr_across_subjects), 
        'psnr_std': np.std(psnr_across_subjects),
        'ssim_mean': np.mean(ssim_across_subjects),
        'ssim_std': np.std(ssim_across_subjects),
        'mae_mean': np.mean(mae_across_subjects),
        'mae_std': np.std(mae_across_subjects)
    }
    
    # 保存评估指标
    results = {
        'avg_psnr': avg_metrics['psnr'],
        'avg_ssim': avg_metrics['ssim'],
        'avg_mae': avg_metrics['mae'],
        'input_modalities': test_dataset[0]['input_modalities'],
        'target_modality': test_dataset[0]['missing_modality'],
        'checkpoint_file': checkpoint_file,
        'num_slices': len(test_dataset),
        'per_subject': subject_avg_metrics,
        'subject_statistics': subject_stats  # 添加受试者间的统计信息
    }
    
    # 将评估指标保存为JSON
    with open(os.path.join(vis_dir, 'metrics.json'), 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=4)
    
    # 打印评估结果
    print(f"\n评估结果:")
    print(f"平均PSNR: {avg_metrics['psnr']:.2f}")
    print(f"平均SSIM: {avg_metrics['ssim']:.4f}")
    print(f"平均MAE: {avg_metrics['mae']:.4f}")
    print(f"\n受试者间统计:")
    print(f"PSNR均值±标准差: {subject_stats['psnr_mean']:.2f}±{subject_stats['psnr_std']:.2f}")
    print(f"SSIM均值±标准差: {subject_stats['ssim_mean']:.4f}±{subject_stats['ssim_std']:.4f}")
    print(f"MAE均值±标准差: {subject_stats['mae_mean']:.4f}±{subject_stats['mae_std']:.4f}")
    print(f"可视化完成，结果保存在: {vis_dir}")
    
    # 创建汇总报告
    with open(os.path.join(vis_dir, 'README.md'), 'w') as f:
        f.write(f"# 可视化结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 模型信息\n\n")
        f.write(f"- 权重文件: `{checkpoint_file}`\n")
        f.write(f"- 输入模态: {test_dataset[0]['input_modalities']}\n")
        f.write(f"- 目标模态: {test_dataset[0]['missing_modality']}\n\n")
        
        f.write(f"## 评估指标\n\n")
        f.write(f"- 平均PSNR: {avg_metrics['psnr']:.2f}\n")
        f.write(f"- 平均SSIM: {avg_metrics['ssim']:.4f}\n")
        f.write(f"- 平均MAE: {avg_metrics['mae']:.4f}\n")
        f.write(f"- 测试切片数量: {len(test_dataset)}\n\n")
        
        f.write(f"## 受试者间统计\n\n")
        f.write(f"- PSNR均值±标准差: {subject_stats['psnr_mean']:.2f}±{subject_stats['psnr_std']:.2f}\n")
        f.write(f"- SSIM均值±标准差: {subject_stats['ssim_mean']:.4f}±{subject_stats['ssim_std']:.4f}\n")
        f.write(f"- MAE均值±标准差: {subject_stats['mae_mean']:.4f}±{subject_stats['mae_std']:.4f}\n\n")
        
        f.write(f"## 目录结构\n\n")
        f.write(f"在每个受试者的目录下:\n")
        f.write(f"- `slice_XXXX_output.png` - 生成的图像\n")
        f.write(f"- `slice_XXXX_target.png` - 真实目标图像\n")
        f.write(f"- `metrics.json` - 详细评估指标\n")
        
        # 添加每个受试者的指标
        f.write(f"\n## 受试者级别指标\n\n")
        f.write(f"| 受试者ID | PSNR | SSIM | MAE |\n")
        f.write(f"|---------|------|------|-----|\n")
        for subj, subj_avg in subject_avg_metrics.items():
            f.write(f"| {subj} | {subj_avg['psnr']:.2f} | {subj_avg['ssim']:.4f} | {subj_avg['mae']:.4f} |\n")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='MRI模态合成结果可视化')
    parser.add_argument('--data_path', type=str, default="../dataset/IXI-Aligned-DP130_80_SMt2",
                        help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/t1=t2',
                        help='模型保存路径，默认为 ./checkpoint/t1=t2')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help='指定要加载的checkpoint文件名，默认使用最新的checkpoint')
    parser.add_argument('--input_modalities', type=str, nargs='+', 
                        help='指定输入模态，例如：--input_modalities t1')
    parser.add_argument('--target_modality', type=str,
                        help='指定目标模态，例如：--target_modality t2')
    parser.add_argument('--gpu', type=int, default=0,
                        help='指定使用的GPU ID，默认为0')
    
    args = parser.parse_args()
    
    # 设置GPU设备
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {args.gpu}")
    
    # 直接使用args而不是创建新的config字典
    visualize_results(args) 