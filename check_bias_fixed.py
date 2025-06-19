from logging import BufferingFormatter
import os
import glob
import wave
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torch import Tensor
from model.pretrained.dual_head_cnn14 import DualHeadCnn14Simple
from predict import preprocess_audio as load_audio  # reuse the same logic
    

def get_logits(model, files, device):
    logits = []
    for file_path in tqdm(files, desc="Evaluating"):
        try:
            waveform = load_audio(file_path=file_path).to(device)  # [1, T]
            print(f"Initial waveform shape: {waveform.shape}")
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)  # [1, 1, T]
            elif waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1:
                pass  # already correct
            else:
                print(f"❌ Unexpected input shape: {waveform.shape}")
                continue

            with torch.no_grad():
                input_tensor = waveform  # expected by model
                print(f"{file_path} → waveform shape: {input_tensor.shape}")
                binary_logit, _ = model(input_tensor)
                logit = binary_logit.squeeze().item()
                prob = torch.sigmoid(binary_logit).squeeze().item()
    
                print(f"  → Logit: {logit:.4f}, Sigmoid Prob: {prob:.4f}")
                logits.append(logit)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
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

state_dict = torch.load("/Users/sumerjoshi/upwork/audio-processing-ai/model/saved_models/Cnn14_16k_mAP_around2000_samplingAndRealTransformChanges_20250615_0746.pth", map_location=device)
model = DualHeadCnn14Simple(pretrained=False)  # Use False since weights are from your training
model.load_state_dict(state_dict)
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