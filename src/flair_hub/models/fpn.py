import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Build a Swin-Unet-like skip hierarchy from standard ViT token features.

    The SMP Unet decoder used with Swin expects six encoder outputs:
    [input, dummy_h2, h4, h8, h16, h32].
    A plain ViT only provides H/16 token grids, so this module synthesizes the
    skip hierarchy with learned projections and resize blocks.
    """

    def __init__(
        self,
        in_channels: int = 768,
        fusion_channels: int = 256,
        stage_channels: Sequence[int] = (128, 256, 512, 1024),
    ) -> None:
        super().__init__()

        if len(stage_channels) != 4:
            raise ValueError(f"Expected four stage channels, got {len(stage_channels)}")

        self.in_channels = in_channels
        self.fusion_channels = fusion_channels
        self.stage_channels = list(stage_channels)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, fusion_channels, kernel_size=1) for _ in range(4)]
        )

        self.c2_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, stage_channels[0], kernel_size=3, padding=1),
        )
        self.c3_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, stage_channels[1], kernel_size=3, padding=1),
        )
        self.c4_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, stage_channels[2], kernel_size=3, padding=1),
        )
        self.c5_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.GELU(),
            nn.Conv2d(fusion_channels, stage_channels[3], kernel_size=3, padding=1),
        )

    def _reshape_patch_features_to_spatial(self, patch_features: torch.Tensor) -> torch.Tensor:
        bsz, num_patches, embed_dim = patch_features.shape
        spatial_size = int(math.sqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square number")
        return patch_features.view(bsz, spatial_size, spatial_size, embed_dim).permute(0, 3, 1, 2)

    def forward(self, image: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"Expected 4 ViT stages, got {len(features)}")

        stages = [self._reshape_patch_features_to_spatial(feat) for feat in features]
        p1, p2, p3, p4 = [proj(stage) for proj, stage in zip(self.lateral_convs, stages)]

        # Fuse transformer depth information at the native ViT resolution.
        t4 = p4
        t3 = p3 + t4
        t2 = p2 + t3
        t1 = p1 + t2

        c4 = self.c4_refine(t4)  # H/16

        c3 = F.interpolate(t3, scale_factor=2, mode="bilinear", align_corners=False)
        c3 = self.c3_refine(c3)  # H/8

        c2 = F.interpolate(t2 + t1, scale_factor=4, mode="bilinear", align_corners=False)
        c2 = self.c2_refine(c2)  # H/4

        c5 = self.c5_refine(t4)  # H/32

        bsz = image.shape[0]
        h_half, w_half = image.shape[-2] // 2, image.shape[-1] // 2
        dummy = image.new_empty((bsz, 0, h_half, w_half))

        return [image, dummy, c2, c3, c4, c5]


class ViT_FPN(nn.Module):
    """
    ViT backbone wrapped with a pyramid neck that matches SMP Swin-Unet outputs.
    """

    def __init__(
        self,
        vit_encoder: nn.Module,
        fusion_channels: int = 256,
        stage_channels: Sequence[int] = (128, 256, 512, 1024),
    ) -> None:
        super().__init__()

        self.vit_encoder = vit_encoder
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.vit_encoder.embed_dim,
            fusion_channels=fusion_channels,
            stage_channels=stage_channels,
        )
        self.out_channels = [self.vit_encoder.in_channels, 0, *stage_channels]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        vit_features = self.vit_encoder(x)
        print(len(vit_features))
        return self.fpn(x, vit_features)
