from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from typing import Tuple
from zipfile import ZipFile
import glob
from torch_audiomentations import Compose

class AIAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=10.0, train: bool = True, ai_transform: Compose = None) -> None:
        """
        Args:
            root_dir: str  Path to .mp3 and .wav files to create the training.
            sample_rate: int  set to 16khz for training
            duration: float   set to 10 seconds for sampling
        """
        real_files = glob.glob("data/train/real/**/*.wav", recursive=True) + \
             glob.glob("data/train/real/**/*.mp3", recursive=True)
        ai_files = glob.glob("data/train/ai/**/*.wav", recursive=True) + \
                glob.glob("data/train/ai/**/*.mp3", recursive=True)
                
        self.audio_files = real_files + ai_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(sample_rate * duration)
        self.train = train
        self.labels = []
        self.ai_transform = ai_transform
        
        for f in self.audio_files:
            parent = str(Path(f).parent).lower()
            if "ai" in parent.split("/"):
                self.labels.append(1.0)
            else:
                self.labels.append(0.0)
                
        print(f"Train Set To: {self.train}")
        print(f"AI Transform Set To: {self.ai_transform}")


    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        afile_path = self.audio_files[idx]
        label = self.labels[idx]
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
            
        # Ai Transform only for AI Labels
        if label == 1:
            waveform = waveform.unsqueeze(0) 
            waveform = self.ai_transform(waveform, sample_rate=16000)
            waveform = waveform.squeeze(0)
                        
        if self.train:
            waveform += 0.001 * torch.randn_like(waveform)

        # Return as shape: (samples,)
        return waveform.squeeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)