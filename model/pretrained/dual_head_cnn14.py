import torch.nn as nn
from .cnn14 import Cnn14

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
