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

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def train_model(data_folder: str, num_epochs: int, saved_path: str) -> None:
    dataset = AIAudioDataset(data_folder)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DualHeadCnn14(pretrained=True)

    # 500-2000 file range used for training. unfreezing things here.
    for name, param in model.named_parameters():
        if any(block in name for block in ["conv_block6", "fc_binary"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    loss_fn = nn.BCEWithLogitsLoss()

    # only optimize trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    train_loop(
        model=model,
        loader=loader,
        num_epochs=num_epochs,
        saved_path=saved_path,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )


def train_loop(
    model: DualHeadCnn14,
    loader: DataLoader,
    num_epochs: int,
    saved_path: str,
    optimizer: Adam,
    loss_fn: BCEWithLogitsLoss,
) -> None:
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for each_input, labels in loader:
            labels = labels.float().unsqueeze(1)
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
    parser = argparse.ArgumentParser(
        prog="train.py", description="Python File to finetune audio files"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="Number of epochs for finetuning"
    )
    parser.add_argument(
        "--dataFolder",
        type=dir_path,
        default="data/train/",
        help="Data to Load to Train",
    )
    parser.add_argument(
        "--savedPath",
        help="Needs to be a file path to a .pth file to save the model",
        required=True,
    )
    args = parser.parse_args()

    dataFolder = args.dataFolder
    num_epochs = args.num_epochs
    saved_path = args.savedPath
    base_name, extension = os.path.splitext(saved_path)
    new_saved_path = f"{base_name}_{timestamp}{extension}"
    train_model(dataFolder, num_epochs, saved_path=new_saved_path)
