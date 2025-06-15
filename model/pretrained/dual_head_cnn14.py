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

        self.fc_audioset = self.fc_audioset

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