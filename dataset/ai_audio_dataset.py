from pathlib import Path
import torch
import torchaudio
import random
from typing import Tuple
from zipfile import ZipFile

class AIAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=10.0) -> None:
        """
        Args:
            root_dir: str  Path to .mp3 and .wav files to create the training. 
            sample_rate: int  set to 16khz for training
            duration: float   set to 10 seconds for sampling
        """
        self.paths = list(Path(root_dir).rglob("*.wav")) + list(Path(root_dir).rglob("*.mp3"))
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_len = int(sample_rate * duration)

    
    def __len__(self) -> int:
        return len(self.paths)
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)

        total_len = waveform.shape[1]

        if total_len < self.audio_len:
            pad_len = self.audio_len - total_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            start = random.randint(0, total_len - self.audio_len)
            waveform = waveform[:, start:start + self.audio_len]

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            f_min=50,
            f_max=8000
        )(waveform)

        logmel = torch.log(mel_spec + 1e-6)

        label = 1 if "ai" in str(path).lower() else 0
        return logmel.squeeze(0), torch.tensor(label)