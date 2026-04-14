import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ViTAdapter(nn.Module):
    """
    Adapter for raw ViT intermediate outputs.

    Expected input is a tuple of:
    1. the original image tensor shaped (B, C, H, W)
    2. a list/tuple of four raw ViT feature tensors, each shaped
       (B, num_patches, embed_dim)

    The adapter forwards the real image unchanged and converts the four token
    tensors into a spatial global-token map shaped (B, out_channels, H_p, W_p)
    for the Local Patch Refiner.
    """

    def __init__(
        self,
        in_channels=3072,
        out_channels=512,
        use_checkpoint=True,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.channel_reducer = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels)
        )

    def _forward_impl(self, f1, f2, f3, f4):
        concat_features = torch.cat([f1, f2, f3, f4], dim=-1)
        return self.channel_reducer(concat_features)

    def forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 2:
            raise ValueError("ViTAdapter expects (image, raw_vit_features)")

        image, vit_features = features
        if len(vit_features) != 4:
            raise ValueError(f"ViTAdapter expects 4 raw ViT feature tensors, got {len(vit_features)}")

        f1, f2, f3, f4 = vit_features

        ref_shape = f1.shape
        for idx, feat in enumerate((f2, f3, f4), start=2):
            if feat.shape[:2] != ref_shape[:2]:
                raise ValueError(
                    f"Feature {idx} shape {feat.shape} is incompatible with feature 1 shape {ref_shape}"
                )

        if self.use_checkpoint and any(f.requires_grad for f in [f1, f2, f3, f4]):
            reduced_tokens = checkpoint(self._forward_impl, f1, f2, f3, f4, use_reentrant=False)
        else:
            reduced_tokens = self._forward_impl(f1, f2, f3, f4)

        bsz, num_patches, _ = reduced_tokens.shape
        spatial_size = int(num_patches ** 0.5)
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Token count {num_patches} is not a square number")

        global_tokens = reduced_tokens.view(bsz, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)

        return image, global_tokens
