import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from dataset import BRATSDataset

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data loading - initialize dataset first to determine input channels
        self.dataset = BRATSDataset(config['data_path'], 'train', config['input_modalities'], config['force_target_modality'])
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'])
            
        # Initialize validation dataset
        self.val_dataset = BRATSDataset(config['data_path'], 'val', config['input_modalities'], config['force_target_modality'])
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=config['num_workers'])
        
        # Determine input channels - get from first sample
        sample = self.dataset[0]
        in_channels = sample['input'].shape[0]
        
        # Initialize models - use dynamically determined input channels
        self.G = Generator(in_channels=in_channels).to(self.device)
        self.D = Discriminator(in_channels=in_channels + 1).to(self.device)  # +1 because discriminator input is concatenation of source and target/generated images
        
        # Learning rate configuration
        self.lr_g = config['lr_g']
        self.lr_d = config['lr_d']
        self.lr_decay_epoch = config['lr_decay_epoch']
        self.lr_decay_factor = config['lr_decay_factor']
        
        # Optimizers
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr_g, betas=(0.5, 0.999), weight_decay=1e-5)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr_d, betas=(0.5, 0.999), weight_decay=1e-5)
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=self.lr_decay_epoch, gamma=self.lr_decay_factor)
        self.scheduler_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=self.lr_decay_epoch, gamma=self.lr_decay_factor)
        
        # Loss functions
        self.criterion_adv = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        
        # Other configurations
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Record best validation results
        self.best_val_psnr = 0
        self.best_epoch = 0

    def validate(self, epoch):
        """Evaluate model performance on validation set"""
        self.G.eval()
        metrics = {'psnr': [], 'ssim': [], 'mae': []}
        
        print(f"Evaluating model on validation set...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation Progress"):
                real_A = batch['input'].to(self.device)
                real_B = batch['target'].to(self.device)
                
                # Generate prediction
                fake_B = self.G(real_A)
                
                # Calculate MAE (Mean Absolute Error)
                mae = torch.abs(fake_B - real_B).mean().item()
                metrics['mae'].append(mae)
                
                # Convert to numpy for metric calculation
                real_B_np = real_B.cpu().numpy()
                fake_B_np = fake_B.cpu().numpy()
                
                # Calculate PSNR and SSIM
                for i in range(real_B_np.shape[0]):
                    metrics['psnr'].append(peak_signal_noise_ratio(real_B_np[i], fake_B_np[i]))
                    try:
                        ssim_value = structural_similarity(
                            real_B_np[i, 0], fake_B_np[i, 0],
                            data_range=1.0,
                            win_size=min(7, min(real_B_np.shape[2], real_B_np.shape[3]) - 1),
                            multichannel=False
                        )
                        metrics['ssim'].append(ssim_value)
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        metrics['ssim'].append(0.0)
        
        # Calculate average metrics
        avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ssim = sum(metrics['ssim']) / len(metrics['ssim'])
        avg_mae = sum(metrics['mae']) / len(metrics['mae'])
        
        print(f"Validation Results - Epoch {epoch}:")
        print(f"  PSNR: {avg_psnr:.2f}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  MAE:  {avg_mae:.4f}")
        
        # Check if this is the best model
        is_best = avg_psnr > self.best_val_psnr
        if is_best:
            self.best_val_psnr = avg_psnr
            self.best_epoch = epoch
            
            # Save best model
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'G_state_dict': self.G.state_dict(),
                'D_state_dict': self.D.state_dict(),
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'mae': avg_mae,
                'input_modalities': self.dataset[0]['input_modalities'],
                'target_modality': self.dataset[0]['missing_modality']
            }, best_model_path)
            print(f"Saved new best model (PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.4f})")
        
        return {'psnr': avg_psnr, 'ssim': avg_ssim, 'mae': avg_mae}

    def train_epoch(self, epoch):
        self.G.train()
        self.D.train()
        
        total_loss_G = 0
        total_loss_D = 0
        batch_count = 0
        
        pbar = tqdm(self.train_loader, 
                    desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        for batch in pbar:
            real_A = batch['input'].to(self.device)
            real_B = batch['target'].to(self.device)
            
            # Train discriminator
            self.opt_D.zero_grad()
            fake_B = self.G(real_A)
            
            # Add noise to discriminator input - helps stabilize GAN training
            noise_real = torch.randn_like(real_B) * 0.05
            noise_fake = torch.randn_like(fake_B) * 0.05
            
            # Add noise to discriminator input
            pred_real = self.D(torch.cat([real_A, real_B + noise_real], dim=1))
            pred_fake = self.D(torch.cat([real_A, fake_B.detach() + noise_fake], dim=1))
            
            loss_D_real = self.criterion_adv(pred_real, torch.ones_like(pred_real))
            loss_D_fake = self.criterion_adv(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.opt_D.step()
            
            # Train generator
            self.opt_G.zero_grad()
            # Generator forward pass without noise
            pred_fake = self.D(torch.cat([real_A, fake_B], dim=1))
            loss_G_adv = self.criterion_adv(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = self.criterion_l1(fake_B, real_B) * 5
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            self.opt_G.step()
            
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}",
                'D_loss': f"{loss_D.item():.4f}"
            })
        
        # Update learning rates
        self.scheduler_G.step()
        self.scheduler_D.step()
        current_lr_g = self.scheduler_G.get_last_lr()[0]
        current_lr_d = self.scheduler_D.get_last_lr()[0]
        
        # Print training information
        avg_loss_G = total_loss_G / batch_count
        avg_loss_D = total_loss_D / batch_count
        print(f"Epoch {epoch}: G_loss={avg_loss_G:.4f}, D_loss={avg_loss_D:.4f}, lr_G={current_lr_g:.6f}, lr_D={current_lr_d:.6f}")
        print(f"Input modalities: {self.dataset[0]['input_modalities']}, Target modality: {self.dataset[0]['missing_modality']}")
            
        # Modified checkpoint saving logic
        # Save at epoch 1 or every 5 epochs
        if (epoch + 1) == 1 or (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'G_state_dict': self.G.state_dict(),
                'D_state_dict': self.D.state_dict(),
                'opt_G': self.opt_G.state_dict(),
                'opt_D': self.opt_D.state_dict(),
                'scheduler_G': self.scheduler_G.state_dict(),
                'scheduler_D': self.scheduler_D.state_dict(),
                'input_modalities': self.dataset[0]['input_modalities'],
                'target_modality': self.dataset[0]['missing_modality']
            }, os.path.join(self.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))
            print(f"Saved checkpoint: checkpoint_{epoch+1}.pth")
        
        # Validate after each epoch
        val_metrics = self.validate(epoch)
        
        # Print best results information
        print(f"Current best model: Epoch {self.best_epoch}, PSNR={self.best_val_psnr:.2f}")
        
        return avg_loss_G, avg_loss_D, val_metrics 