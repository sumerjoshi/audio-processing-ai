import os
import glob
import wave
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from model.pretrained.dual_head_cnn14 import DualHeadCnn14

def load_audio(path, sample_rate=16000, target_length=64000):
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    waveform = waveform.mean(dim=0, keepdim=True)  # mono: [1, T]

    # Ensure the input has at least target_length (e.g., 4s of audio)
    if waveform.shape[1] < target_length:
        pad_len = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_length]  # truncate if too long

    return waveform  # [1, target_length]

def get_logits(model, files, device):
    logits = []
    for file in tqdm(files, desc="Evaluating"):
        try:
            waveform = load_audio(file, target_length=64000).to(device)
            waveform = waveform.to(device)  # waveform: [1, T]
            waveform = waveform.unsqueeze(0)       # [1, 1, T]
            waveform = waveform.unsqueeze(2)       # [1, 1, 1, T]
            
            if waveform.ndim != 4:
                
                print(f"❌ Invalid input shape {waveform.shape} → expected [1, 1, 1, T]")
                continue
            
            with torch.no_grad():
                print(f"{file} → waveform shape: {waveform.shape}")
                logit = model(waveform).squeeze().item()
                logits.append(logit)
                
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    return logits

def safe_mean(x):
    return np.mean(x) if len(x) > 0 else float("nan")

# Gather files (wav + mp3) from data/train
real_files = glob.glob("data/train/real/**/*.wav", recursive=True) + \
             glob.glob("data/train/real/**/*.mp3", recursive=True)
ai_files = glob.glob("data/train/ai/**/*.wav", recursive=True) + \
           glob.glob("data/train/ai/**/*.mp3", recursive=True)

print(f"🟩 Found {len(real_files)} real audio files.")
print(f"🟥 Found {len(ai_files)} AI audio files.")

if not real_files:
    print("⚠️ Warning: No real files found.")
if not ai_files:
    print("⚠️ Warning: No AI files found.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DualHeadCnn14(pretrained=True)
model.load_state_dict(torch.load("/Users/sumerjoshi/upwork/audio-processing-ai/model/saved_models/Cnn14_16k_mAP_around2000_20250614_1223.pth", map_location=device))
model.eval()
model.to(device)

real_logits = get_logits(model, real_files, device)
ai_logits = get_logits(model, ai_files, device)

avg_real_logit = safe_mean(real_logits)
avg_ai_logit = safe_mean(ai_logits)

print("\n=== Bias Check Results ===")
if real_logits:
    print(f"Real  Logit Avg: {avg_real_logit:.4f} | Sigmoid: {torch.sigmoid(torch.tensor(avg_real_logit)).item():.4f}")
else:
    print("⚠️ No real logits computed.")

if ai_logits:
    print(f"AI    Logit Avg: {avg_ai_logit:.4f} | Sigmoid: {torch.sigmoid(torch.tensor(avg_ai_logit)).item():.4f}")
else:
    print("⚠️ No AI logits computed.")