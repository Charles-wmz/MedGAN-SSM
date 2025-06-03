import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BRATSDataset(Dataset):
    def __init__(self, base_path, phase='train', input_modalities=None, force_target_modality=None):
        self.data_path = os.path.join(base_path, phase)
        self.subjects = [d for d in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, d))]
        
        # User-specified input and target modalities
        self.input_modalities = input_modalities  # If None, use all non-missing modalities
        self.force_target_modality = force_target_modality  # If None, use missing modality from missing_modality.txt
        
        # Validate modality names
        self.all_modalities = ['t1', 't1ce', 't2', 'flair', 'pd']
        if self.input_modalities:
            for mod in self.input_modalities:
                if mod not in self.all_modalities:
                    raise ValueError(f"Invalid input modality: {mod}. Valid modalities are: {self.all_modalities}")
        
        if self.force_target_modality and self.force_target_modality not in self.all_modalities:
            raise ValueError(f"Invalid target modality: {self.force_target_modality}. Valid modalities are: {self.all_modalities}")
            
        # Check for conflicts between input and target modalities
        if self.input_modalities and self.force_target_modality:
            if self.force_target_modality in self.input_modalities:
                raise ValueError(f"Target modality {self.force_target_modality} cannot be used as input modality")
    
    def __len__(self):
        return len(self.subjects) * 80  # Assuming 80 slices per volume
    
    def __getitem__(self, idx):
        # Calculate actual subject index and slice index
        slice_per_volume = 80
        subj_idx = idx // slice_per_volume
        slice_idx = idx % slice_per_volume
        
        subj = self.subjects[subj_idx]
        # Load modality data
        modalities = {}
        
        # Read missing modality information - for reference only
        missing_from_file = []
        missing_file_path = os.path.join(self.data_path, subj, 'missing_modality.txt')
        if os.path.exists(missing_file_path):
            with open(missing_file_path) as f:
                missing_from_file = f.read().splitlines()
        
        # Determine target modality
        if self.force_target_modality:
            # If user specified target modality, it is considered "missing" (to be predicted)
            missing_modality = self.force_target_modality
        elif missing_from_file:
            # Otherwise use missing modality from file
            missing_modality = missing_from_file[0]
        else:
            # Default missing modality
            missing_modality = 'flair'
        
        # Load all available modality data
        available_modalities = []
        for mod in self.all_modalities:
            try:
                path = os.path.join(self.data_path, subj, f'{mod}.npy')
                if os.path.exists(path):
                    volume = np.load(path)
                    # Extract specific slice
                    modalities[mod] = volume[slice_idx]
                    available_modalities.append(mod)
            except Exception:
                # Silently ignore unloadable modalities
                pass
        
        # Determine input modalities
        if self.input_modalities:
            # User specified input modalities, check which are available
            input_mods = [mod for mod in self.input_modalities if mod in modalities and mod != missing_modality]
            
            # If no specified input modalities are available, use all available non-target modalities
            if not input_mods:
                input_mods = [mod for mod in available_modalities if mod != missing_modality]
        else:
            # Use all available non-target modalities as input
            input_mods = [mod for mod in available_modalities if mod != missing_modality]
        
        # Ensure at least one input modality
        if not input_mods:
            # If really no input modalities available, use random noise
            noise_mod = np.random.randn(256, 256).astype(np.float32)  # Assuming 256x256 image size
            noise_mod = (noise_mod - noise_mod.min()) / (noise_mod.max() - noise_mod.min()) * 2 - 1  # Normalize to [-1,1]
            modalities['noise'] = noise_mod
            input_mods = ['noise']
        
        # Build input data
        input_data = np.stack([modalities[mod] for mod in input_mods], axis=0)
        
        # Load target data
        target = None
        
        # First try loading from original modalities directory
        original_path = os.path.join(self.data_path, subj, 'original_modalities', f'original_{missing_modality}.npy')
        if os.path.exists(original_path):
            try:
                volume = np.load(original_path)
                target = volume[slice_idx]
            except Exception:
                pass
        
        # If not in original directory, use current directory modality data
        if target is None and missing_modality in modalities:
            target = modalities[missing_modality]
        
        # If target still not available, use random noise
        if target is None:
            target = np.random.randn(256, 256).astype(np.float32)
            target = (target - target.min()) / (target.max() - target.min()) * 2 - 1
        
        return {
            'input': torch.FloatTensor(input_data),
            'target': torch.FloatTensor(target).unsqueeze(0),  # Add channel dimension
            'missing_modality': missing_modality,
            'input_modalities': input_mods
        } 