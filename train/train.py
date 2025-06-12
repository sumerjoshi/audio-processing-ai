import argparse
import os
import datetime
from dataset import ai_audio_dataset
from model.pretrained.cnn14 import Cnn14
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import logging

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
save_path = f"model/saved_models/final_model_{timestamp}.pth"

class DualHeadCnn14(Cnn14):
    def __init__(self, pretrained=True):
          super().__init__(pretrained=pretrained)

          self.fc_audioset = self.fc_audioset

          self.fc_binary = nn.Linear(2048, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, mel_bins, time_steps)
        
        Returns:
            binary_logit: shape(B,1)
            tag_logits: shape (B, 527)
        """
        x = self.extract_features(x)
        tag_logits = self.fc_audioset(x)
        binary_logit = self.fc_binary(x)
        return binary_logit, tag_logits



def train_model(dataFolder: str, epochs: int, saved_path: str = save_path) -> None:
    dataset = ai_audio_dataset.AIAudioDataset(dataFolder)
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

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
                labels = labels.float().unsqueeze(1)
                binary_logits, _ = model(inputs)
                loss = loss_fn(binary_logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()

        logging.DEBUG(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}") 

    torch.save(model.state_dict(), saved_path)
    logging.INFO(f"Model saved to {saved_path}")


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
    args = parser.parse_args()

    dataFolder = args.dataFolder
    epochs = args.epoch
    train_model(dataFolder, epochs)

