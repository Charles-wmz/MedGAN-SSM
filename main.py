import os
import torch
import argparse
from datetime import datetime

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from dataset import BRATSDataset
from trainer import Trainer
from tester import Tester

def parse_args():
    parser = argparse.ArgumentParser(description='MedGAN-SSM Training and Testing')
    
    # Basic settings
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'test', 'both'],
                        help='Run mode: train, test, or both')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint_file', type=str, default='best_model.pth',
                        help='Specific checkpoint file to load')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=2e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--lr_decay_epoch', type=int, default=40,
                        help='Epoch to start learning rate decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Learning rate decay factor')
    
    # Modality settings
    parser.add_argument('--input_modalities', nargs='+', default=['t1', 't2'],
                        help='Input modalities to use')
    parser.add_argument('--target_modality', type=str, default='flair',
                        help='Target modality to synthesize')
    
    # Other settings
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training mode
    if args.mode in ['train', 'both']:
        print(f"\n=== Starting Training ===")
        print(f"Input modalities: {args.input_modalities}")
        print(f"Target modality: {args.target_modality}")
        
        # Initialize trainer
        trainer = Trainer({
            'data_path': args.data_path,
            'checkpoint_dir': args.checkpoint_dir,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr_g': args.lr_g,
            'lr_d': args.lr_d,
            'lr_decay_epoch': args.lr_decay_epoch,
            'lr_decay_factor': args.lr_decay_factor,
            'input_modalities': args.input_modalities,
            'force_target_modality': args.target_modality,
            'num_workers': args.num_workers
        })
        
        # Start training
        for epoch in range(args.epochs):
            trainer.train_epoch(epoch)
    
    # Testing mode
    if args.mode in ['test', 'both']:
        print(f"\n=== Starting Testing ===")
        print(f"Input modalities: {args.input_modalities}")
        print(f"Target modality: {args.target_modality}")
        
        # Initialize tester
        tester = Tester({
            'data_path': args.data_path,
            'checkpoint_dir': args.checkpoint_dir,
            'checkpoint_path': os.path.join(args.checkpoint_dir, args.checkpoint_file),
            'input_modalities': args.input_modalities,
            'force_target_modality': args.target_modality,
            'num_workers': args.num_workers
        })
        
        # Run testing
        results = tester.evaluate()
        
        # Print results
        print("\nTest Results:")
        print(f"PSNR: {results['metrics']['psnr']:.2f}")
        print(f"SSIM: {results['metrics']['ssim']:.4f}")
        print(f"MAE:  {results['metrics']['mae']:.4f}")

if __name__ == '__main__':
    main()