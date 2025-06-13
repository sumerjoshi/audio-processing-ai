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
from typing import Literal
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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

    mel_spec = MelSpectrogram(
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
            "filename", "is_ai_generated", "ai_confidence",
            "top_tag_1", "tag_1_confidence",
            "top_tag_2", "tag_2_confidence",
            "top_tag_3", "tag_3_confidence",
            "top_tag_4", "tag_4_confidence",
            "top_tag_5", "tag_5_confidence"
        ])

def append_row(csv_path: str, file_path: str, ai_label: Literal['Yes', 'No'], ai_prob: Tensor, top5_tags: list[Tuple[str, float]]) -> None:
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        row = [file_path, ai_label, f"{ai_prob:.3f}"]
        for tag, conf in top5_tags:
            row.extend([tag, f"{conf:.3f}"])
        writer.writerow(row)


def predict_folder(folder_path: str, model_path: str, csv_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualHeadCnn14(pretrained=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    write_header(csv_path)

    audio_files_to_test = list(Path(folder_path).rglob("*.wav")) + list(Path(folder_path).rglob("*.mp3"))

    for file_path in tqdm(audio_files_to_test, desc="Running Prediction"):
        file_path = str(file_path)
        input_tensor = preprocess_audio(file_path=file_path).to(device=device)
        ai_prob, tag_probs = predict_one(model, input_tensor)

        ai_label = "Yes" if ai_prob > 0.5 else "No"
        top5_idx = tag_probs.argsort()[-5:][::-1]
        top5_tags = [(AUDIOSCENE_TAGS[i], float(tag_probs[i])) for i in top5_idx]

        append_row(csv_path, Path(file_path).name, ai_label, ai_prob, top5_tags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing .mp3/.wav files to predict against", required=True)
    parser.add_argument("--model",help="Trained model in model/pretrained/saved_models/ or your own trained model", required=True)
    args = parser.args()
    
    csv_path = f"predictions_{timestamp}.csv"
    predict_folder(args.folder, args.model, csv_path)
