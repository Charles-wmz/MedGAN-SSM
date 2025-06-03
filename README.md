# MedGAN-SSM: Medical Image Synthesis with State Space Modeling

A deep learning framework for synthesizing missing MRI modalities using a novel State Space Modeling approach.

## Overview

This project implements a GAN-based framework for synthesizing missing MRI modalities from available ones. The model uses a novel State Space Modeling approach to capture both local and global features in medical images, resulting in high-quality synthetic images.

## Features

- Dynamic handling of multiple input modalities
- State Space Modeling for better feature representation
- Multi-scale feature extraction
- Dynamic gating mechanism
- Non-local attention for capturing long-range dependencies
- Comprehensive evaluation metrics (PSNR, SSIM, MAE)

## Project Structure

```
MedGAN-SSM/
├── src/
│   ├── models/
│   │   ├── generator.py      # Generator model with State Space Module
│   │   └── discriminator.py  # PatchGAN-based discriminator
│   ├── data/
│   │   └── dataset.py        # BRATS dataset loader
│   ├── training/
│   │   └── trainer.py        # Training logic
│   ├── testing/
│   │   └── tester.py         # Testing and evaluation
│   └── main.py              # Main program
├── checkpoint/              # Model checkpoints
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- scikit-image
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/main.py --mode train \
    --data_path /path/to/dataset \
    --input_modalities t1 t2 \
    --target_modality flair \
    --batch_size 1 \
    --epochs 60
```

### Testing

```bash
python src/main.py --mode test \
    --data_path /path/to/dataset \
    --checkpoint_file best_model.pth \
    --input_modalities t1 t2 \
    --target_modality flair
```

### Training and Testing

```bash
python src/main.py --mode both \
    --data_path /path/to/dataset \
    --input_modalities t1 t2 \
    --target_modality flair
```

## Command Line Arguments

- `--data_path`: Path to the dataset directory
- `--checkpoint_dir`: Directory to save model checkpoints
- `--batch_size`: Batch size for training (default: 1)
- `--epochs`: Number of training epochs (default: 60)
- `--lr_g`: Generator learning rate (default: 2e-4)
- `--lr_d`: Discriminator learning rate (default: 2e-4)
- `--lr_decay_epoch`: Epoch to start learning rate decay (default: 40)
- `--lr_decay_factor`: Learning rate decay factor (default: 0.1)
- `--input_modalities`: Input modalities to use (e.g., t1 t2)
- `--target_modality`: Target modality to synthesize
- `--mode`: Run mode (train/test/both)
- `--checkpoint_file`: Specific checkpoint file to load
- `--num_workers`: Number of data loading workers (default: 16)

## Model Architecture

### Generator
- Encoder with State Space Module
- Multi-scale S6 module for feature processing
- Dynamic gating mechanism
- Decoder with skip connections

### Discriminator
- PatchGAN-based architecture
- Simplified design for better training stability

## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MAE (Mean Absolute Error)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:


## Acknowledgments

- Thanks to the BRATS and IXI dataset providers
- Inspired by various GAN architectures and medical image synthesis works 