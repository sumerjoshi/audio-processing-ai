import torch
from predict import preprocess_audio, predict_one
import os
from model.pretrained.dual_head_cnn14 import DualHeadCnn14

# Manually specify your two known paths
AI_AUDIO_PATH = f"{os.getcwd()}/data/train/ai/auto_gen_songs/song_C349.wav"     # Replace with actual file
REAL_AUDIO_PATH = f"{os.getcwd()}/data/train/real/fma_extra_small_119_124/124/124154.mp3" # Replace with actual file
MODEL_PATH = f"/Users/sumerjoshi/upwork/audio-processing-ai/model/saved_models/Cnn14_16k_mAP_around2000_unfreezeconv456_fcbinary_20250614_1815.pth"  # Replace with actual model

print(f"AI: AUDIO PATH: {AI_AUDIO_PATH}")
print(f"REAL AUDIO PATH: {REAL_AUDIO_PATH}")
print(f"MODEL PATH: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DualHeadCnn14(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

def debug_prediction(audio_path):
    print(f"\n--- Predicting for {audio_path} ---")
    input_tensor = preprocess_audio(audio_path).to(device)
    with torch.no_grad():
        binary_logit, tag_logits = model(input_tensor)
        sigmoid_prob = torch.sigmoid(binary_logit)
        print(f"Raw binary logit: {binary_logit.item():.6f}")
        print(f"Sigmoid probability: {sigmoid_prob.item():.6f}")
        print(f"Top tag indices: {tag_logits.squeeze().topk(5).indices.tolist()}")
        print(f"Top tag confidences: {torch.sigmoid(tag_logits).squeeze().topk(5).values.tolist()}")

debug_prediction(AI_AUDIO_PATH)
debug_prediction(REAL_AUDIO_PATH)