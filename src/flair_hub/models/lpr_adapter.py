import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LPRAdapter(nn.Module):
    def __init__(self, in_channels=1920, out_channels=768, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.channel_reducer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def _forward_impl(self, f1, f2, f3, f4):
        # Dynamically map to the f3 resolution, preventing cat mismatches for non-512 sizes
        target_size = f3.shape[-2:]
        
        f1_proj = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2_proj = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3_proj = f3 
        f4_proj = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        
        concat_features = torch.cat([f1_proj, f2_proj, f3_proj, f4_proj], dim=1)
        
        return self.channel_reducer(concat_features)

    def forward(self, features):
        img = features[0]
        
        # Robustly extract the 4 main backbone feature maps. (SMP usually outputs 6 tensors for swin base)
        if len(features) >= 6:
            f1, f2, f3, f4 = features[-4:]
        else:
            # Fallback if there's a different configuration
            f1, f2, f3, f4 = features[1], features[2], features[3], features[4]
        
        # Checkpoint the interpolation and concat block to save VRAM 
        if self.use_checkpoint and any(f.requires_grad for f in [f1, f2, f3, f4]):
            reduced_features = checkpoint(self._forward_impl, f1, f2, f3, f4, use_reentrant=False)
        else:
            reduced_features = self._forward_impl(f1, f2, f3, f4)
        
        return img, reduced_features