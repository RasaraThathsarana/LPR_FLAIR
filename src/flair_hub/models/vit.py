import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, List

from .fpn import ViT_FPN

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm library not found.")

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("torchvision.models not found.")


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that mimics the interface of segmentation_models_pytorch encoders.
    Provides raw transformer features at multiple depths.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        in_channels: int = 3,
        img_size: int = 512,
        pretrained: bool = True,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.use_checkpoint = use_checkpoint
        self.model_name = model_name

        # Create the base ViT model
        if TIMM_AVAILABLE and model_name in timm.list_models():
            # Use timm for more ViT variants
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_channels,
                img_size=img_size,
                num_classes=0,  # Remove classification head
                global_pool=''  # No global pooling
            )
        elif TORCHVISION_AVAILABLE and hasattr(tv_models, model_name.replace('_', '')):
            # Fallback to torchvision ViT
            model_func = getattr(tv_models, model_name.replace('_', ''))
            self.vit = model_func(
                weights='DEFAULT' if pretrained else None,
                num_classes=0
            )
            # Adapt input channels if needed
            if in_channels != 3:
                self.vit.conv_proj = nn.Conv2d(
                    in_channels, self.vit.conv_proj.out_channels,
                    kernel_size=self.vit.conv_proj.kernel_size,
                    stride=self.vit.conv_proj.stride,
                    padding=self.vit.conv_proj.padding
                )
        else:
            raise ValueError(f"Model {model_name} not found in timm or torchvision")

        # Extract ViT components
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

        # Get model dimensions
        self.embed_dim = self.vit.embed_dim
        self.depth = len(self.blocks)
        self.num_heads = self.blocks[0].attn.num_heads
        self.patch_size = self.patch_embed.patch_size[0]  # Get first element if tuple

        # Calculate number of patches
        self.num_patches = (img_size // self.patch_size) ** 2

        # Define output channels at different stages
        self.out_channels = [
            self.embed_dim,  # After block 3
            self.embed_dim,  # After block 6
            self.embed_dim,  # After block 9
            self.embed_dim,  # After final block
        ]

    def _forward_impl(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store features at different stages
        features = []

        # Process through transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            # Extract raw features at stages 3, 6, 9, and Final (i is 0-indexed)
            if i == 2:  # After Block 3
                features.append(x[:, 1:, :])
            elif i == 5:  # After Block 6
                features.append(x[:, 1:, :])
            elif i == 8:  # After Block 9
                features.append(x[:, 1:, :])
            elif i == self.depth - 1:  # Final Layer
                features.append(x[:, 1:, :])

        return features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class FLAIR_ViT(nn.Module):
    """
    FLAIR ViT model wrapper that matches the interface of FLAIR_Monotemp.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        channels: int = 3,
        classes: int = 19,
        img_size: int = 512,
        return_type: str = 'encoder'
    ) -> None:
        super().__init__()

        self.return_type = return_type
        assert self.return_type in ['encoder', 'decoder'], \
            'return_type should be one of ["encoder", "decoder"]'

        # Get ViT configuration from config
        vit_config = config.get('models', {}).get('vit_model', {})
        model_name = vit_config.get('model_name', 'vit_base_patch16_224')
        pretrained = vit_config.get('pretrained', True)
        fusion_channels = vit_config.get('fusion_channels', 256)
        stage_channels = vit_config.get('stage_channels', [128, 256, 512, 1024])

        # Read the gradient checkpointing config flag
        use_checkpoint = config.get('models', {}).get('use_gradient_checkpointing', False)

        vit_encoder = ViTEncoder(
            model_name=model_name,
            in_channels=channels,
            img_size=img_size,
            pretrained=pretrained,
            use_checkpoint=use_checkpoint
        )

        if self.return_type == 'encoder':
            self.seg_model = ViT_FPN(
                vit_encoder=vit_encoder,
                fusion_channels=fusion_channels,
                stage_channels=stage_channels,
            )
        else:
            raise NotImplementedError(
                "FLAIR_ViT is encoder-only. Decoder mode should use the standard SMP decoder path."
            )
