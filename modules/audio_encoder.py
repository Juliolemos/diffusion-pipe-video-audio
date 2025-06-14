import torch.nn as nn

class SimpleAudioEncoder(nn.Module):
    def __init__(self, mel_bins=80, embed_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, mel_bins)),
            nn.Flatten(1),
            nn.Linear(64 * mel_bins, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, mel):
        return self.net(mel)  # (B, embed_dim)
