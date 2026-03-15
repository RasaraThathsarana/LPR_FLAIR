import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from torch.cuda.amp import autocast

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        mid_channels = out_channels // self.expansion

        # 1×1 reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3×3 spatial conv
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1×1 expansion
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # projection if shape mismatch
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def _forward_impl(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out += identity
        return F.relu(out, inplace=True)

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

class UNet_FullRes(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Encoder
        self.enc1 = BasicBlock(in_channels, base_channels, use_checkpoint=use_checkpoint)
        self.enc2 = BasicBlock(base_channels, base_channels * 2, stride=2, use_checkpoint=use_checkpoint)
        self.enc3 = BasicBlock(base_channels * 2, base_channels * 4, stride=2, use_checkpoint=use_checkpoint)
        self.enc4 = BasicBlock(base_channels * 4, base_channels * 8, stride=2, use_checkpoint=use_checkpoint)
        
        # Bottleneck
        self.bottleneck = BasicBlock(base_channels * 8, base_channels * 16, stride=2, use_checkpoint=use_checkpoint)
        
        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = BasicBlock(base_channels * 16 + base_channels * 8, base_channels * 8, use_checkpoint=use_checkpoint)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = BasicBlock(base_channels * 8 + base_channels * 4, base_channels * 4, use_checkpoint=use_checkpoint)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = BasicBlock(base_channels * 4 + base_channels * 2, base_channels * 4, use_checkpoint=use_checkpoint)
        
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_refine = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        
        out = self.final_refine(torch.cat([self.final_up(d2), s1], dim=1))
        
        return out

class LocalPatchRefiner(nn.Module):
    def __init__(self, global_dim, in_channels=3, patch_size=16, hidden_dim=256, cnn_dim=32, num_heads=8, use_checkpoint=True, warmup_iters=15000, ramp_iters=5000, local_backbone='unet'):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        self.cnn = UNet_FullRes(in_channels=in_channels, base_channels=cnn_dim, use_checkpoint=use_checkpoint)

        # Dimension calculation for Query: out(cnn_dim*4)
        combined_dim = cnn_dim * 4
        
        self.query_proj = nn.Sequential(
            nn.Conv2d(combined_dim, cnn_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.channel_meanings = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(global_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(global_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 2D Sine-Cosine Positional Encoding for spatial awareness within the patch
        pos_embed = self._get_2d_sincos_pos_embed(hidden_dim, patch_size)
        self.register_buffer('pos_embed', pos_embed)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        # embed_dim must be even
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        pos_embed = torch.cat([emb_h, emb_w], dim=1) # (grid_size*grid_size, embed_dim)
        return pos_embed.unsqueeze(0) # (1, grid_size*grid_size, embed_dim)

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000**omega)
        
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        
        return torch.cat([emb_sin, emb_cos], dim=1)

    def _attention_block(self, q_normed, k_normed, v_normed):
    # Force fp32 for attention computation to maintain numerical stability
        with autocast(enabled=False):

            q_normed = q_normed.float()
            k_normed = k_normed.float()
            v_normed = v_normed.float()
            
            Q = self.q_proj(q_normed)
            K = self.k_proj(k_normed)
            V = self.v_proj(v_normed)

            attn_logits = (Q @ K.t()) * (self.hidden_dim ** -0.5)
            attn_weights = torch.tanh(attn_logits)

            x_attn = attn_weights * V

            out = self.out_proj(x_attn)

        return out

    def forward(self, img, global_tokens):
        B, C, H, W = img.shape
        P = self.patch_size
        
        out_feat = self.cnn(img)
        all_feats = out_feat

        n_h, n_w = H // P, W // P

        # Robust check to ensure global_tokens match spatial dims of img//P
        if global_tokens.shape[-2] != n_h or global_tokens.shape[-1] != n_w:
            global_tokens = F.interpolate(global_tokens, size=(n_h, n_w), mode='bilinear', align_corners=False)

        global_tokens = global_tokens.permute(0, 2, 3, 1) # [B, n_h, n_w, global_dim]

        # Patching for Query construction
        patches_q = all_feats.unfold(2, P, P).unfold(3, P, P)
        patches_q = patches_q.permute(0, 2, 3, 1, 4, 5).reshape(-1, all_feats.size(1), P, P)

        if self.use_checkpoint and patches_q.requires_grad:
            q_map = checkpoint(self.query_proj, patches_q, use_reentrant=False)
        else:
            q_map = self.query_proj(patches_q)
            
        q_normed = self.q_norm(q_map.flatten(2).transpose(1, 2))
        
        # Inject 2D Sine-Cosine Positional Encodings
        q_normed = q_normed + self.pos_embed

        k_normed = self.k_norm(self.channel_meanings)
        v_normed = self.v_norm(global_tokens).reshape(B * n_h * n_w, 1, self.global_dim)

        if self.use_checkpoint and q_normed.requires_grad:
            attn_features = checkpoint(self._attention_block, q_normed, k_normed, v_normed, use_reentrant=False)
        else:
            attn_features = self._attention_block(q_normed, k_normed, v_normed)
        
        # Prepare components for Fusion (only attention features now)
        fusion_input = attn_features.transpose(1, 2).reshape(-1, self.hidden_dim, P, P)

        if self.use_checkpoint and fusion_input.requires_grad:
            fused = checkpoint(self.fusion_conv, fusion_input, use_reentrant=False)
        else:
            fused = self.fusion_conv(fusion_input)

        out = fused.view(B, n_h, n_w, self.hidden_dim, P, P)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, self.hidden_dim, H, W)
        
        return out