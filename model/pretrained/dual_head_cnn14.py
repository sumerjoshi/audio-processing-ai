import torch
from .cnn14 import Cnn14


class DualHeadCnn14(Cnn14):
    def __init__(
        self,
        sample_rate=16000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
        pretrained=True,
    ):
        super().__init__(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            classes_num,
            pretrained=pretrained,
        )

        self.fc1 = torch.nn.Linear(2048, 2048)
        self.dropout = torch.nn.Dropout(p=0.5)

        #self.fc_audioset = self.fc_audioset

        self.fc_binary = torch.nn.Linear(2048, 1)

        self.bn0 = torch.nn.BatchNorm2d(1)

    def forward(self, x):
        """
        Args:
            x: (B, 32000)

        Returns:
            binary_logit: shape(B,1)
            tag_logits: shape (B, 527)
        """
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.spectrogram_extractor(x)

        x = self.logmel_extractor(x)

        x = x.transpose(2, 3)

        x = self.bn0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = torch.mean(x, dim=3)  # time avg
        x = x.mean(dim=2)  # freq avg
        x = self.fc1(x)
        x = self.dropout(x)

        tag_logits = self.fc_audioset(x)
        binary_logit = self.fc_binary(x)
        return binary_logit, tag_logits
    

import torch
from .cnn14 import Cnn14


class DualHeadCnn14Simple(Cnn14):
    def __init__(
        self,
        sample_rate=16000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
        pretrained=True,
    ):
        super().__init__(
            sample_rate,
            window_size,
            hop_size,
            mel_bins,
            fmin,
            fmax,
            classes_num,
            pretrained=pretrained,
        )

        # Binary classification head with progressive dimension reduction
        self.fc_binary_1 = torch.nn.Linear(2048, 1024)  
        self.dropout_binary_1 = torch.nn.Dropout(p=0.6)
        
        self.fc_binary_2 = torch.nn.Linear(1024, 512)  
        self.dropout_binary_2 = torch.nn.Dropout(p=0.5)
        
        self.fc_binary_final = torch.nn.Linear(512, 1) 
        
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU(inplace=True)

        # AudioSet head remains unchanged (inherited fc_audioset)
        # It will use the original 2048-dim features

    def forward(self, x):
        """
        Args:
            x: (B, 32000)

        Returns:
            binary_logit: shape(B,1)
            tag_logits: shape (B, 527)
        """
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(2, 3)
        x = self.bn0(x)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = torch.mean(x, dim=3)  # time avg
        x = x.mean(dim=2)  # freq avg
        # Now x is shape (B, 2048)
        
        # AudioSet head: use original 2048-dim features directly
        tag_logits = self.fc_audioset(x)
        
        # Binary classification head: progressive dimension reduction
        binary_features = self.fc_binary_1(x)
        binary_features = self.relu(binary_features)
        binary_features = self.dropout_binary_1(binary_features)
        
        binary_features = self.fc_binary_2(binary_features)
        binary_features = self.relu(binary_features)
        binary_features = self.dropout_binary_2(binary_features)
        
        binary_logit = self.fc_binary_final(binary_features)
        
        return binary_logit, tag_logits