import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.models.generator import Generator
from dataset import BRATSDataset

class Tester:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = config['checkpoint_dir']  # Save checkpoint directory path
        
        # Initialize dataset first to determine input channels
        self.dataset = BRATSDataset(config['data_path'], 'test', config['input_modalities'], config['force_target_modality'])
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config['num_workers'])
            
        # Determine input channels - get from first sample
        sample = self.dataset[0]
        in_channels = sample['input'].shape[0]
        
        # Initialize model - use dynamically determined input channels
        self.G = Generator(in_channels=in_channels).to(self.device)
        self.load_checkpoint(config['checkpoint_path'])
        
    def load_checkpoint(self, path):
        self.checkpoint_path = path  # Save checkpoint path
        try:
            # Method 1: Remove weights_only=True
            checkpoint = torch.load(path)
            
            # Get epoch information
            self.epoch = checkpoint.get('epoch', -1)  # Return -1 if no epoch information
            
            # Check if checkpoint contains modality information, print as reference only, no warning
            if 'input_modalities' in checkpoint and 'target_modality' in checkpoint:
                saved_input_mods = checkpoint['input_modalities']
                saved_target_mod = checkpoint['target_modality']
                current_input_mods = self.dataset[0]['input_modalities']
                current_target_mod = self.dataset[0]['missing_modality']
                
                # Print modality information as reference
                print(f"Modality information in checkpoint: Input={saved_input_mods}, Target={saved_target_mod}")
                print(f"Current modality configuration: Input={current_input_mods}, Target={current_target_mod}")
            
            # Print epoch information from checkpoint
            if 'epoch' in checkpoint:
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            
            # Try loading state dictionary, handle possible channel mismatch
            try:
                self.G.load_state_dict(checkpoint['G_state_dict'])
                print("Successfully loaded model parameters")
            except RuntimeError as e:
                print(f"Model structure mismatch with checkpoint, performing partial loading...")
                
                # Get current model and saved model state dictionaries
                model_dict = self.G.state_dict()
                pretrained_dict = checkpoint['G_state_dict']
                
                # Filter layers with matching names, ignore size mismatches
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                
                # Update current model state dictionary
                model_dict.update(pretrained_dict)
                self.G.load_state_dict(model_dict)
                print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers")
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
        
    def evaluate(self):
        metrics = {'psnr': [], 'ssim': [], 'mae': []}
        
        print(f"Testing...")
        print(f"Input modalities: {self.dataset[0]['input_modalities']}, Target modality: {self.dataset[0]['missing_modality']}")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing Progress"):
                real_A = batch['input'].to(self.device)
                real_B = batch['target'].cpu().numpy()
                fake_B = self.G(real_A).cpu().numpy()
                
                for i in range(fake_B.shape[0]):
                    metrics['psnr'].append(float(peak_signal_noise_ratio(real_B[i], fake_B[i])))  # Convert to Python float
                    
                    try:
                        ssim_value = structural_similarity(
                            real_B[i, 0], fake_B[i, 0],
                            data_range=1.0,
                            win_size=min(7, min(real_B.shape[2], real_B.shape[3]) - 1),
                            multichannel=False
                        )
                        metrics['ssim'].append(float(ssim_value))  # Convert to Python float
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        print(f"Image shapes: real_B={real_B.shape}, fake_B={fake_B.shape}")
                        metrics['ssim'].append(0.0)
                    
                    metrics['mae'].append(float(np.abs(real_B[i] - fake_B[i]).mean()))  # Convert to Python float
        
        # Add more information to results
        result = {
            'epoch': self.epoch,
            'test_info': {
                'input_modalities': self.dataset[0]['input_modalities'],
                'target_modality': self.dataset[0]['missing_modality'],
                'checkpoint_file': os.path.basename(self.checkpoint_path)
            },
            'metrics': {
                'psnr': float(np.mean(metrics['psnr'])),
                'ssim': float(np.mean(metrics['ssim'])),
                'mae': float(np.mean(metrics['mae']))
            }
        }
        
        # Add timestamp
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use correct path to save results
        results_filename = f'results_{self.epoch}.json'
        results_path = os.path.join(self.checkpoint_dir, results_filename).replace('\\', '/')
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {results_path}")
        
        return result 