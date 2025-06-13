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

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_path = f"model/saved_models/final_model_{timestamp}.pth"


def train_model(dataFolder: str, epochs: int, saved_path: str = save_path) -> None:
    dataset = AIAudioDataset(dataFolder)
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_loop(model, loader, optimizer, epochs, loss_fn, saved_path)

def train_loop(model: DualHeadCnn14, loader: DataLoader, epochs: int, saved_path: str, optimizer: Adam, loss_fn: BCEWithLogitsLoss) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for each_input, labels in loader:
                print(f"type(each_input): {type(each_input)}")
                labels = labels.float().unsqueeze(1)
                binary_logits, _ = model(each_input)
                loss = loss_fn(binary_logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()


        logging.debug(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}") 

    torch.save(model.state_dict(), saved_path)
    logging.info(f"Model saved to {saved_path}")


def dir_path(string) -> str:
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Python File to finetune audio files"
    )
    parser.add_argument('--epoch', type=int, default=5, help='Epoch is needed for finetuning')
    parser.add_argument('--dataFolder', type=dir_path, default="data/train/", help='Data to Load to Train')
    parser.add_argument('--savePath', help="Needs to be a file path to a .pth file to save the model", required=True)
    args = parser.parse_args()

    dataFolder = args.dataFolder
    epochs = args.epoch
    savePath = args.savePath
    train_model(dataFolder, epochs, saved_path=savePath)

