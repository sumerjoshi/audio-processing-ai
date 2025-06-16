import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple, Literal
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torchaudio.transforms import MelSpectrogram
from tqdm.auto import tqdm
from model.pretrained.dual_head_cnn14 import DualHeadCnn14Simple
import pandas as pd 
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
AUDIOSCENE_TAGS = [
    line.strip()
    for line in open("inference/audioset_labels_no_index.txt", encoding="utf-8")
]
DEFAULT_CSV_PATH = f"predictions_{timestamp}.csv"


def preprocess_audio(file_path, sample_rate=16000, duration=10.0) -> Tensor:
    waveform, sr = torchaudio.load(file_path)

    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform_len = waveform.shape[1]
    target_len = int(sample_rate * duration)

    if waveform_len < target_len:
        pad = target_len - waveform_len
        waveform = F.pad(waveform, (0, pad))
    else:
        start = (waveform_len - target_len) // 2
        waveform = waveform[:, start : start + target_len]

    return waveform

def predict_one(model, input_tensor: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.squeeze(0)
        input_tensor = input_tensor.unsqueeze(0)
        binary_logit, tag_logits = model(input_tensor)
        ai_prob = torch.sigmoid(binary_logit).item()
        tag_probs = torch.sigmoid(tag_logits).squeeze().cpu().numpy()
    return ai_prob, tag_probs


def write_header(csv_path):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "is_ai_generated",
                "ai_confidence",
                "top_tag_1",
                "tag_1_confidence",
                "top_tag_2",
                "tag_2_confidence",
                "top_tag_3",
                "tag_3_confidence",
                "top_tag_4",
                "tag_4_confidence",
                "top_tag_5",
                "tag_5_confidence",
            ]
        )


def append_row(
    csv_path: str,
    file_path: str,
    ai_label: Literal["Yes", "No"],
    ai_prob: Tensor,
    top5_tags: list[Tuple[str, float]],
) -> None:
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [file_path, ai_label, f"{ai_prob:.3f}"]
        for tag, conf in top5_tags:
            row.extend([tag, f"{conf:.3f}"])
        writer.writerow(row)

def get_audio_files(folder_path: str) -> list[Path]:
    """Get all audio files with case-insensitive extension matching"""
    folder = Path(folder_path)
    audio_files = []
    
    # Supported audio extensions (case-insensitive)
    supported_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    
    # Find all files and check extensions case-insensitively
    for file_path in folder.rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            audio_files.append(file_path)
    
    return audio_files
        
def write_final_accuracy_row(csv_path: str) -> str:
    """Write summary statistics to Excel file"""
    base_name = csv_path.replace('.csv', '')
    xlsx_path = f"{base_name}.xlsx"
    
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate summary statistics
        total_files = len(df)
        ai_generated_count = len(df[df['is_ai_generated'] == 'Yes'])
        real_count = len(df[df['is_ai_generated'] == 'No'])
        avg_ai_confidence = df['ai_confidence'].mean()
        
        # Write to Excel
        df.to_excel(xlsx_path, index=False)
        
        try:
            from openpyxl import load_workbook
            from openpyxl.styles import PatternFill
            
            # Load workbook to add summary
            wb = load_workbook(xlsx_path)
            ws = wb.active
            
            # Add summary rows
            ws.append([])  # Blank row
            ws.append(['SUMMARY'])
            ws.append(['Total Files', total_files])
            ws.append(['Predicted AI Generated', ai_generated_count])
            ws.append(['Predicted Real', real_count])
            ws.append(['Average AI Confidence', f"{avg_ai_confidence:.3f}"])
            
            # Highlight summary section
            yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            for row in range(ws.max_row - 4, ws.max_row + 1):
                for col in range(1, 3):  # First two columns
                    ws.cell(row=row, column=col).fill = yellow_fill
            
            wb.save(xlsx_path)
            
        except ImportError:
            print("Note: openpyxl not available for Excel formatting. Basic Excel file created.")
            
        return xlsx_path
        
    except Exception as e:
        print(f"Warning: Could not create Excel file: {e}")
        return csv_path
    
def predict_folder(folder_path: str, model_path: str, csv_path: str, threshold: float = 0.35):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed: Use correct model class
    model = DualHeadCnn14Simple(pretrained=False)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    write_header(csv_path)

    # Fixed: Use case-insensitive file finding
    audio_files_to_test = get_audio_files(folder_path)
    
    print(f"Found {len(audio_files_to_test)} audio files to process")
    
    # Show file types found
    extensions = {}
    for file_path in audio_files_to_test:
        ext = file_path.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print("File types found:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count} files")

    for file_path in tqdm(audio_files_to_test, desc="Running Prediction"):
        try:
            input_tensor = preprocess_audio(file_path=str(file_path)).to(device=device)
            ai_prob, tag_probs = predict_one(model, input_tensor)

            ai_label = "Yes" if ai_prob > threshold else "No"
            
            top5_idx = tag_probs.argsort()[-5:][::-1]
            top5_tags = [(AUDIOSCENE_TAGS[i], float(tag_probs[i])) for i in top5_idx]

            append_row(csv_path, file_path.name, ai_label, ai_prob, top5_tags)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue
    
    # Create Excel file with summary
    xlsx_path = write_final_accuracy_row(csv_path)
    print(f"Results saved to: {csv_path}")
    if xlsx_path != csv_path:
        print(f"Excel file created: {xlsx_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        help="Folder containing .mp3/.wav/.flac/.m4a files to predict against",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Trained model in model/pretrained/saved_models/ or your own trained model",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Output Directory of where to put CSV file",
        required=True
    )
    args = parser.parse_args()
    csv_full_path = f"{args.output}/{DEFAULT_CSV_PATH}"
    

    predict_folder(args.folder, args.model, csv_full_path)