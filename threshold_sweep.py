import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model.pretrained.dual_head_cnn14 import DualHeadCnn14Simple
from predict import preprocess_audio
from pathlib import Path
from tqdm import tqdm
import os
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def predict_logits(model, file_paths, device):
    model.to(device)
    model.eval()
    
    logits = []
    labels = []
    
    for path in tqdm(file_paths):
        label = 1 if "ai" in str(path).lower() else 0
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # make mono

        # Pad or trim to 10s (160000 samples)
        num_target_samples = 160000
        if waveform.shape[1] < num_target_samples:
            padding = num_target_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > num_target_samples:
            waveform = waveform[:, :num_target_samples]

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]
        elif waveform.ndim != 2:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

        # Ensure [1, 1, num_samples]
        input_tensor = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            print(f"Tensor shape before model: {input_tensor.shape}")
            binary_logit, _ = model(input_tensor.squeeze(1))
            logits.append(binary_logit.item())
            labels.append(label)

    return logits, labels

def evaluate_thresholds(logits, labels, thresholds=np.linspace(0.0, 1.0, 50)):
    from scipy.special import expit  # sigmoid
    probs = expit(logits)
    
    print("Threshold | Accuracy | Precision | Recall | F1")
    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        print(f"{t:.2f}     | {acc:.3f}    | {prec:.3f}     | {rec:.3f}  | {f1:.3f}")
        

def plot_logit_distributions(logits, labels):
    ai_logits = [logits[i] for i in range(len(logits)) if labels[i] == 1]
    real_logits = [logits[i] for i in range(len(logits)) if labels[i] == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(ai_logits, bins=50, alpha=0.6, label='AI', density=True)
    plt.hist(real_logits, bins=50, alpha=0.6, label='Real', density=True)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Logit Distribution: AI vs. Real")
    plt.xlabel("Logit Value (before sigmoid)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    MODEL_PATH = f"/Users/sumerjoshi/upwork/audio-processing-ai/model/saved_models/Cnn14_16k_mAP_around2000_samplingAndRealTransformChanges_20250615_0746.pth" 
    model = DualHeadCnn14Simple(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    
    AI_FOLDER = f"{os.getcwd()}/data/train/ai"
    REAL_FOLDER = f"{os.getcwd()}/data/train/real"

    # Load a mixed sample of AI and real audio
    files = list(Path(AI_FOLDER).rglob("*.wav")) + \
            list(Path(REAL_FOLDER).rglob("*.mp3"))
                
    logits, labels = predict_logits(model, files, device="cpu")
    evaluate_thresholds(logits, labels)
    plot_logit_distributions(logits, labels)
    
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    
    probs = torch.sigmoid(logits)

    # Use threshold = 0.5
    threshold = 0.5
    preds = (probs > threshold).int()

    print(f"\n=== Classification Report (Threshold = {threshold}) ===")
    print(classification_report(labels.numpy(), preds.numpy(), labels=[0, 1], target_names=["Real", "AI"]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(labels.numpy(), preds.numpy()))