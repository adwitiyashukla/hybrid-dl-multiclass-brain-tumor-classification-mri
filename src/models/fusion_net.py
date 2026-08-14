from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .cbam import CBAM


class HandcraftedEncoder(nn.Module):

    def __init__(self, n_features: int, hidden: int = 256, out_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):

    def __init__(self, cnn_dim: int, hand_dim: int, out_dim: int = 256):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, out_dim)
        self.hand_proj = nn.Linear(hand_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim + hand_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, f_cnn: torch.Tensor, f_hand: torch.Tensor):
        gate = self.gate(torch.cat([f_cnn, f_hand], dim=1))
        fused = gate * self.cnn_proj(f_cnn) + (1.0 - gate) * self.hand_proj(f_hand)
        return fused, gate


class HybridTumorNet(nn.Module):

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        n_classes: int = 4,
        in_chans: int = 3,
        n_handcrafted: int = 43,
        use_handcrafted: bool = True,
        use_cbam: bool = True,
        pretrained: bool = True,
        dropout: float = 0.3,
        fusion_dim: int = 256,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required. Install it with: pip install timm"
            ) from exc

        self.use_handcrafted = use_handcrafted
        self.use_cbam = use_cbam
        self.n_classes = n_classes

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )
        cnn_dim = self.backbone.num_features

        self.cbam = CBAM(cnn_dim) if use_cbam else None
        self.pool = nn.AdaptiveAvgPool2d(1)

        if use_handcrafted:
            self.handcrafted_encoder = HandcraftedEncoder(
                n_handcrafted, out_dim=128, dropout=dropout
            )
            self.fusion = GatedFusion(cnn_dim, 128, fusion_dim)
            head_dim = fusion_dim
        else:
            self.handcrafted_encoder = None
            self.fusion = None
            head_dim = cnn_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_dim, n_classes),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if self.cbam is not None:
            features = self.cbam(features)
        return features

    def forward(
        self,
        images: torch.Tensor,
        handcrafted: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fmap = self.forward_features(images)
        f_cnn = self.pool(fmap).flatten(1)

        gate_summary = None
        if self.use_handcrafted:
            if handcrafted is None:
                raise ValueError(
                    "model was built with use_handcrafted=True but forward() "
                    "received handcrafted=None"
                )
            f_hand = self.handcrafted_encoder(handcrafted)
            fused, gate = self.fusion(f_cnn, f_hand)
            gate_summary = gate.mean(dim=1)
            logits = self.head(fused)
        else:
            logits = self.head(f_cnn)

        return {
            "logits": logits,
            "features": f_cnn,
            "gate": gate_summary,
            "fmap": fmap,
        }


def build_model(config: dict) -> HybridTumorNet:
    model_cfg = config.get("model", {})
    return HybridTumorNet(
        backbone=model_cfg.get("backbone", "efficientnet_b0"),
        n_classes=model_cfg.get("n_classes", 4),
        in_chans=model_cfg.get("in_chans", 3),
        n_handcrafted=model_cfg.get("n_handcrafted", 43),
        use_handcrafted=model_cfg.get("use_handcrafted", True),
        use_cbam=model_cfg.get("use_cbam", True),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        fusion_dim=model_cfg.get("fusion_dim", 256),
    )
