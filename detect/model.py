"""
UNet with Transformer bottleneck — Transformer-CNN hybrid for semantic segmentation.

Architecture:
  Encoder (CNN):  Conv blocks with downsampling (4 levels)
  Bottleneck:     Transformer encoder (multi-head self-attention + FFN)
  Decoder (CNN):  Upsample + skip connections + Conv blocks (4 levels)
  Output:         1×1 conv → num_classes

Designed for small datasets with active learning — lightweight but expressive.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool → ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
#  Transformer bottleneck
# ---------------------------------------------------------------------------
class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for spatial feature maps."""

    def __init__(self, d_model, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.row_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.col_embed.weight, -0.1, 0.1)

    def forward(self, x):
        """x: (B, C, H, W) → positional encoding (1, C, H, W)."""
        H, W = x.shape[2], x.shape[3]
        rows = self.row_embed(torch.arange(H, device=x.device))  # (H, C//2)
        cols = self.col_embed(torch.arange(W, device=x.device))  # (W, C//2)
        pos = torch.cat([
            rows.unsqueeze(1).expand(-1, W, -1),  # (H, W, C//2)
            cols.unsqueeze(0).expand(H, -1, -1),   # (H, W, C//2)
        ], dim=-1)  # (H, W, C)
        return pos.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)


class TransformerBottleneck(nn.Module):
    """Transformer encoder applied to flattened spatial feature maps.

    Input:  (B, C, H, W)  — CNN feature map
    Output: (B, C, H, W)  — refined feature map with global context
    """

    def __init__(self, d_model, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = PositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        # Add positional encoding
        pos = self.pos_enc(x)
        x = x + pos

        # Flatten spatial dims: (B, C, H, W) → (B, H*W, C)
        x = x.flatten(2).permute(0, 2, 1)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


# ---------------------------------------------------------------------------
#  UNet-Transformer model
# ---------------------------------------------------------------------------
class TransformerUNet(nn.Module):
    """UNet with Transformer bottleneck for semantic segmentation.

    Args:
        in_channels:  Number of input channels (3 for RGB).
        num_classes:  Number of output classes (including background).
        base_ch:      Base channel count (doubled at each encoder level).
        t_heads:      Number of attention heads in transformer.
        t_layers:     Number of transformer encoder layers.
        t_ff:         Transformer feedforward dimension.
        dropout:      Dropout rate in transformer.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=3,
        base_ch=64,
        t_heads=8,
        t_layers=4,
        t_ff=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        ch = base_ch  # 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch)       # → ch
        self.enc2 = DownBlock(ch, ch * 2)             # → ch*2
        self.enc3 = DownBlock(ch * 2, ch * 4)         # → ch*4
        self.enc4 = DownBlock(ch * 4, ch * 8)         # → ch*8

        # Bottleneck: downsample then transformer
        self.bottleneck_down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(ch * 8, ch * 16),               # → ch*16
        )
        self.transformer = TransformerBottleneck(
            d_model=ch * 16,
            nhead=t_heads,
            num_layers=t_layers,
            dim_feedforward=t_ff,
            dropout=dropout,
        )

        # Decoder
        self.dec4 = UpBlock(ch * 16, ch * 8, ch * 8)
        self.dec3 = UpBlock(ch * 8, ch * 4, ch * 4)
        self.dec2 = UpBlock(ch * 4, ch * 2, ch * 2)
        self.dec1 = UpBlock(ch * 2, ch, ch)

        # Output head
        self.out_conv = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # (B, 64,  H,   W)
        e2 = self.enc2(e1)   # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)   # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)   # (B, 512, H/8, W/8)

        # Bottleneck with transformer
        b = self.bottleneck_down(e4)   # (B, 1024, H/16, W/16)
        b = self.transformer(b)         # global attention

        # Decoder with skip connections
        d4 = self.dec4(b, e4)   # (B, 512, H/8, W/8)
        d3 = self.dec3(d4, e3)  # (B, 256, H/4, W/4)
        d2 = self.dec2(d3, e2)  # (B, 128, H/2, W/2)
        d1 = self.dec1(d2, e1)  # (B, 64,  H,   W)

        return self.out_conv(d1)  # (B, num_classes, H, W)

    def predict_proba(self, x):
        """Return softmax probabilities — used for active learning uncertainty."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def build_model(num_classes=3, img_size=256, base_ch=64, t_heads=8, t_layers=4, t_ff=1024):
    """Factory function to build the TransformerUNet model."""
    return TransformerUNet(
        in_channels=3,
        num_classes=num_classes,
        base_ch=base_ch,
        t_heads=t_heads,
        t_layers=t_layers,
        t_ff=t_ff,
    )
