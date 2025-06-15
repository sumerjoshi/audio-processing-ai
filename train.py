import argparse
import os
from datetime import datetime
from dataset.ai_audio_dataset import AIAudioDataset
from model.pretrained.dual_head_cnn14 import DualHeadCnn14
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import logging
from tqdm.auto import tqdm
import random
import numpy as np 
from torch.utils.data import WeightedRandomSampler
import logging
from torch_audiomentations import Compose, PitchShift, Gain, Shift

ai_augment: Compose = Compose(
    transforms=[
        PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.7, sample_rate=16000, output_type='tensor'),
        Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.3, sample_rate=16000, output_type='tensor'),
        Shift(min_shift=-0.1, max_shift=0.1, p=0.5, sample_rate=16000, output_type='tensor'),
    ]
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def train_model(data_folder: str, num_epochs: int, saved_path: str, resume_path: str = None) -> None:
    dataset = AIAudioDataset(root_dir=data_folder, train=True, ai_transform=ai_augment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sampler = compute_sampler(dataset)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    logging.info("Weigh Mistakes on Underepresented Classes")
    labels = [label for _, label in dataset]
    class_counts = torch.bincount(torch.tensor(labels).long())
    pos_weight = class_counts[0] / class_counts[1]
    
    logging.info("Real count (label 0): %d", class_counts[0].item())
    logging.info("AI count (label 1): %d", class_counts[1].item())
    logging.info("pos_weight = %f", pos_weight)

    if resume_path:
        logging.info(f"Resuming from Checkpoint: {resume_path}")
        model = DualHeadCnn14(pretrained=False)
        model.load_state_dict(torch.load(resume_path, map_location='cpu'))
    else:
        logging.info("Starting Fresh with Pretrained Weights")
        model = DualHeadCnn14(pretrained=True)

    # 500-2000 file range used for training. unfreezing things here.
    for name, param in model.named_parameters():
        if any(block in name for block in ["conv_block6", "fc_binary"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        
    # only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5) 

    train_loop(
        model=model,
        loader=loader,
        num_epochs=num_epochs,
        saved_path=saved_path,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def compute_sampler(dataset):
    logging.info("Starting Sampling")
    labels = [label for _, label in dataset]
    class_counts = torch.bincount(torch.tensor(labels).long())
    weights = 1.0 / class_counts.float()
    sample_weights = [weights[int(label)] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def train_loop(
    model: DualHeadCnn14,
    loader: DataLoader,
    num_epochs: int,
    saved_path: str,
    optimizer: Adam,
    loss_fn: BCEWithLogitsLoss,
) -> None:
    logging.info("Training Model Loop")
    model.train()
    smooth = 0.05
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for each_input, labels in loader:
            labels = labels.float().unsqueeze(1)
            labels = labels * (1 - smooth) + 0.5 * smooth
            binary_logits, _ = model(each_input)
            loss = loss_fn(binary_logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            
            avg_loss = running_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
        epoch_loss = running_loss / len(loader)
        logging.debug(f"Epoch {epoch + 1} finished. Avg Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), saved_path)
    logging.info(f"Model saved to {saved_path}")


def dir_path(string) -> str:
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(
        prog="train.py", description="Python File to finetune audio files"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=8, help="Number of epochs for finetuning"
    )
    parser.add_argument(
        "--dataFolder",
        type=dir_path,
        default="data/train/",
        help="Data to Load to Train",
    )
    parser.add_argument(
        "--savedPath",
        default=None,
        help="Needs to be a file path to a .pth file to save the model",
        required=True,
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a .pth model checkpoint to resuem from"
    )
    parser.add_argument('--log-level', type=str, default='INFO',
                    help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    args = parser.parse_args()

    dataFolder = args.dataFolder
    num_epochs = args.num_epochs
    saved_path = args.savedPath
    resume_path = args.resume_from
    base_name, extension = os.path.splitext(saved_path)
    new_saved_path = f"{base_name}_{timestamp}{extension}"
    numeric_level = getattr(logging, args.log_level.upper(), None)
    
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    train_model(dataFolder, num_epochs, new_saved_path, resume_path)
