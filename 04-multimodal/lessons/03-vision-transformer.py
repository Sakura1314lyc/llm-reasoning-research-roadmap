"""一个保留关键数据流的教学版 Vision Transformer。"""

import torch
from torch import nn


class MiniViT(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 8, d_model: int = 64, classes: int = 10):
        super().__init__()
        self.patch_projection = nn.Conv2d(3, d_model, patch_size, patch_size)
        patch_count = (image_size // patch_size) ** 2
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.randn(1, patch_count + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=4 * d_model, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_projection(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(images.size(0), -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.position_embedding)
        logits = self.classifier(self.norm(encoded[:, 0]))
        return logits, encoded


def main() -> None:
    model = MiniViT()
    logits, encoded = model(torch.randn(2, 3, 32, 32))
    print("Encoder tokens / logits：", encoded.shape, logits.shape)
    assert encoded.shape == (2, 17, 64)
    assert logits.shape == (2, 10)


if __name__ == "__main__":
    main()
