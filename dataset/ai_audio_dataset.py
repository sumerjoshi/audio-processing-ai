from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from typing import Tuple
from zipfile import ZipFile
import glob
from torch_audiomentations import Compose

class AIAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=10.0, real_transform: Compose = None, ai_transform: Compose = None, train: bool = True) -> None:
        """
        Args:
            root_dir: str  Path to folder containing 'real' and 'ai' subfolders
            sample_rate: int  set to 16khz for training
            duration: float   set to 10 seconds for sampling
        """
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.train = train
        self.ai_transform = ai_transform
        self.real_transform = real_transform
        
        # Use the actual root_dir instead of hardcoded paths
        real_files = []
        ai_files = []
        
        # Look for real files in root_dir/real/
        real_folder = self.root_dir / "real"
        if real_folder.exists():
            patterns = ["**/*.wav", "**/*.mp3", "**/*.flac", "**/*.m4a", "**/*.WAV", "**/*.MP3"]
            for pattern in patterns:
                real_files.extend(real_folder.glob(pattern))
        
        # Look for AI files in root_dir/ai/
        ai_folder = self.root_dir / "ai"
        if ai_folder.exists():
            patterns = ["**/*.wav", "**/*.mp3", "**/*.flac", "**/*.m4a", "**/*.WAV", "**/*.MP3"]
            for pattern in patterns:
                ai_files.extend(ai_folder.glob(pattern))
        
        # Build file list and labels together to ensure they match
        self.audio_files = []
        self.labels = []
        
        # Add real files with label 0
        for f in real_files:
            self.audio_files.append(str(f))
            self.labels.append(0.0)
        
        # Add AI files with label 1
        for f in ai_files:
            self.audio_files.append(str(f))
            self.labels.append(1.0)
        
        print(f"Dataset root: {self.root_dir}")
        print(f"Found {len(real_files)} real files")
        print(f"Found {len(ai_files)} AI files")
        print(f"Total files: {len(self.audio_files)}")
        print(f"Train Set To: {self.train}")
        print(f"Real Transform Set To: {self.real_transform}")
        print(f"AI Transform Set To: {self.ai_transform}")
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {self.root_dir}. Check that 'real' and 'ai' subfolders exist with audio files.")

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        afile_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            waveform, sr = torchaudio.load(afile_path)
        except Exception as e:
            print(f"❌ Error loading {afile_path}: {e}")
            # Return a dummy waveform if file fails to load
            waveform = torch.zeros(1, self.target_len)
            sr = self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Mono mix if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Normalize audio
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        else:
            waveform = torch.zeros_like(waveform)

        # Pad or trim to target length
        waveform_len = waveform.shape[1]
        if waveform_len < self.target_len:
            pad = self.target_len - waveform_len
            waveform = F.pad(waveform, (0, pad))
        else:
            # Random crop during training, center crop during inference
            if self.train and waveform_len > self.target_len:
                start = torch.randint(0, waveform_len - self.target_len + 1, (1,)).item()
                waveform = waveform[:, start:start + self.target_len]
            else:
                # Center crop
                start = (waveform_len - self.target_len) // 2
                waveform = waveform[:, start:start + self.target_len]
        
        # Apply transformations based on label
        if self.train and label == 0.0 and self.real_transform is not None:
            # Apply transformations to real audio
            waveform = waveform.unsqueeze(0)  # Add batch dimension for transforms
            waveform = self.real_transform(waveform, sample_rate=self.sample_rate)
            waveform = waveform.squeeze(0)  # Remove batch dimension
        elif self.train and label == 1.0 and self.ai_transform is not None:
            # Apply transformations to AI audio
            waveform = waveform.unsqueeze(0)  # Add batch dimension for transforms
            waveform = self.ai_transform(waveform, sample_rate=self.sample_rate)
            waveform = waveform.squeeze(0)  # Remove batch dimension
        
        # Add small noise during training for regularization
        if self.train:
            noise_level = 0.001
            waveform += noise_level * torch.randn_like(waveform)

        # Return as shape: (samples,) - squeeze out the channel dimension
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.float32)