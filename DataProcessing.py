# -*- coding: utf-8 -*-
'''
Preprocess the IXI dataset with N4 bias correction, Z-score normalization,
and intensity scaling to [-1, 1].
'''

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import random
import shutil

# Dataset path
data_path = r"D:\Project-2\dataset\IXI-Aligned"

# Output path for the preprocessed dataset
output_path = r"D:\Project-2\dataset\IXI-Aligned-DP130"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# N4 bias-field correction
def n4_bias_correction(input_image):
    sitk_image = sitk.ReadImage(input_image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)  # Cast image to 32-bit float
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)

# Z-score normalization
def z_score_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# Scale intensities to [-1, 1]
def intensity_scaling(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * (data - data_min) / (data_max - data_min) - 1

# Dataset split helper
def split_dataset(dataset_path):
    """
    Split dataset into train/validation/test subsets.
    """
    # Create target folders for each split
    train_dir = os.path.join(os.path.dirname(dataset_path), "train")
    val_dir = os.path.join(os.path.dirname(dataset_path), "val")
    test_dir = os.path.join(os.path.dirname(dataset_path), "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Collect all subject IDs
    subject_ids = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    random.shuffle(subject_ids)
    
    # Split ratio: 70% train, 15% validation, 15% test
    train_size = int(len(subject_ids) * 0.8)
    val_size = int(len(subject_ids) * 0.1)
    
    train_ids = subject_ids[:train_size]
    val_ids = subject_ids[train_size:train_size+val_size]
    test_ids = subject_ids[train_size+val_size:]
    
    # Copy data into the corresponding folders
    for subject_id in train_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(train_dir, subject_id))
    
    for subject_id in val_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(val_dir, subject_id))
    
    for subject_id in test_ids:
        shutil.copytree(os.path.join(dataset_path, subject_id), os.path.join(test_dir, subject_id))

# IXI dataset preprocessing
def process_ixi_dataset():
    # Traverse all files within the IXI dataset
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Determine modality type (T1, T2, PD, MRA, DTI)
                modality = None
                subject_id = None
                
                # Parse filename to retrieve modality and subject ID
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
                    continue  # Skip unsupported modalities
                
                # Extract the complete subject ID from the filename (e.g., IXI012-HH-1211)
                file_name_without_ext = file.split('.')[0]  # Remove extension
                subject_parts = file_name_without_ext.rsplit('-', 1)[0]  # Remove modality suffix, keep full ID
                subject_id = subject_parts  # Complete ID, e.g., IXI012-HH-1211
                
                # Absolute file path
                file_path = os.path.join(root, file)
                
                # Create output folder per subject using the complete ID
                subject_output_folder = os.path.join(output_path, subject_id)
                if not os.path.exists(subject_output_folder):
                    os.makedirs(subject_output_folder)
                
                # Skip processing if this modality already exists
                output_file = os.path.join(subject_output_folder, f"{modality}.npy")
                if os.path.exists(output_file):
                    print(f"Skipping {file} as it is already processed.")
                    continue
                
                print(f"Processing {file}...")
                
                try:
                    # Apply preprocessing operations
                    n4_corrected = n4_bias_correction(file_path)
                    normalized = z_score_normalization(n4_corrected)
                    scaled = intensity_scaling(normalized)
                    
                    # Persist the preprocessed data
                    np.save(output_file, scaled)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

# Run IXI preprocessing
process_ixi_dataset()
print("Preprocessing complete.")

# Split dataset based on the preprocessed output path
split_dataset(output_path)
print("Splitting complete.")
