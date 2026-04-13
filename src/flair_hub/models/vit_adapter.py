import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ViTAdapter(nn.Module):
    """
    Adapter for raw ViT intermediate outputs.

    Expected input is a list/tuple of four tensors from the ViT encoder, each
    shaped (B, num_patches, embed_dim). The adapter produces:
    1. an image-like proxy tensor suitable for the Local Patch Refiner CNN
    2. a spatial global-token map shaped (B, out_channels, H_p, W_p)
    """

    def __init__(
        self,
        in_channels=3072,
        out_channels=512,
        image_channels=3,
        patch_size=16,
        use_checkpoint=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.channel_reducer = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels)
        )
        self.image_projector = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, image_channels)
        )

    def _forward_impl(self, f1, f2, f3, f4):
        concat_features = torch.cat([f1, f2, f3, f4], dim=-1)
        reduced_tokens = self.channel_reducer(concat_features)
        image_tokens = self.image_projector(concat_features)
        return reduced_tokens, image_tokens

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(f"ViTAdapter expects 4 raw ViT feature tensors, got {len(features)}")

        f1, f2, f3, f4 = features

        ref_shape = f1.shape
        for idx, feat in enumerate((f2, f3, f4), start=2):
            if feat.shape[:2] != ref_shape[:2]:
                raise ValueError(
                    f"Feature {idx} shape {feat.shape} is incompatible with feature 1 shape {ref_shape}"
                )

        if self.use_checkpoint and any(f.requires_grad for f in [f1, f2, f3, f4]):
            reduced_tokens, image_tokens = checkpoint(self._forward_impl, f1, f2, f3, f4, use_reentrant=False)
        else:
            reduced_tokens, image_tokens = self._forward_impl(f1, f2, f3, f4)

        bsz, num_patches, _ = reduced_tokens.shape
        spatial_size = int(num_patches ** 0.5)
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Token count {num_patches} is not a square number")

        global_tokens = reduced_tokens.view(bsz, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)

        img_proxy = image_tokens.view(bsz, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)
        img_proxy = F.interpolate(
            img_proxy,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False,
        )

        return img_proxy, global_tokens
