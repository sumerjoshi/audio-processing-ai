import argparse
import csv
import numpy as np
import torch
from torch import Tensor
import torchaudio
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path
from torchaudio.transforms import MelSpectrogram
from model.pretrained.dual_head_cnn14 import DualHeadCnn14
import tqdm

AUDIOSCENE_TAGS = [line.strip() for line in open("inference/audioset_class_labels.txt")]

def preprocess_audio(file_path, sample_rate=16000, duration=10.0) -> Tensor:
    waveform, sr = torchaudio.load(file_path)

    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform_len = waveform.shape[1]
    target_len = int(sample_rate*duration)

    if waveform_len < target_len:
        pad = target_len - waveform_len
        waveform = F.pad(waveform, (0, pad))
    else:
        start = np.random.randint(0, waveform_len - target_len + 1)
        waveform = waveform[:, start:start + target_len]

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=320,
        n_mels=64,
        f_min=50,
        f_max=8000
    )(waveform)
    
    logmel = torch.log(mel_spec + 1e-6)
    return logmel.unsqueeze(0)

def predict_one(model, input_tensor: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        binary_logit, tag_logits = model(input_tensor)
        ai_prob = torch.sigmoid(binary_logit).item()
        tag_probs = torch.sigmoid(tag_logits).squeeze().cpu().numpy()
    return ai_prob, tag_probs

def write_header(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.write_row([
            
        ])

def predict_folder(folder_path: str, model_path: str, csv_path="predictions.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualHeadCnn14(pretrained=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    write_header(csv_path)

    audio_files_to_test = list(Path(folder_path).rglob("*.wav")) + list(Path(folder_path).rglob("*.mp3"))

    for file_path in tqdm(audio_files_to_test, desc="Running Prediction"):
        input_tensor = preprocess_audio(str(file_path=file_path)).to(device=device)
        ai_prob, tag_probs = predict_one(model, input_tensor)

        ai_label = "Yes" if ai_prob > 0.5 else "No"
        top5_idx = tag_probs.argsort()[-5:][::-1]
        top5_tags = [(AUDIOSCENE_TAGS[i], float(tag_probs[i])) for i in top5_idx]

        append_row(csv_path, file_path.name, ai_label, ai_prob, top5_tags)


