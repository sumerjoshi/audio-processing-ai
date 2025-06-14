from pathlib import Path
import torch
import torchaudio
import random
import torch.nn.functional as F
from typing import Tuple
from zipfile import ZipFile


class AIAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=10.0, train: bool = True) -> None:
        """
        Args:
            root_dir: str  Path to .mp3 and .wav files to create the training.
            sample_rate: int  set to 16khz for training
            duration: float   set to 10 seconds for sampling
        """
        self.audio_files = list(Path(root_dir).rglob("*.wav")) + list(
            Path(root_dir).rglob("*.mp3")
        )
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        
        self.labels = [1.0 if "ai" in str(Path(f).parent).lower() else 0.0 for f in self.audio_files]


    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        afile_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(afile_path)
        
       # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Mono mix if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform = waveform / waveform.abs().max()

        # Pad or trim
        waveform_len = waveform.shape[1]
        if waveform_len < self.target_len:
            pad = self.target_len - waveform_len
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_len]
            
        if self.train:
            waveform += 0.001 * torch.randn_like(waveform)

        # Return as shape: (samples,)
        return waveform.squeeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)