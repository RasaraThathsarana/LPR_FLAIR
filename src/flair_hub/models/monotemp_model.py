import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.checkpoint import checkpoint

from typing import Dict, Any

class DecoderWrapper(nn.Module):
    """Handles sequential execution of the decoder and segmentation head."""
    
    def __init__(self, decoder: nn.Module, segmentation_head: nn.Module, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.decoder = decoder
        self.segmentation_head = segmentation_head
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, *features: Any) -> torch.Tensor:
        decoder_output = self.decoder(*features)  
        return self.segmentation_head(decoder_output)

    def forward(self, *features: Any) -> torch.Tensor:
        # Apply gradient checkpointing if enabled and gradients are required
        if self.use_checkpoint and any(isinstance(f, torch.Tensor) and f.requires_grad for f in features):
            return checkpoint(self._forward_impl, *features, use_reentrant=False)
        return self._forward_impl(*features)

class FLAIR_Monotemp(nn.Module):
    """Monotemporal FLAIR model for segmentation."""
    
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

        encoder, decoder = config['models']['monotemp_model']['arch'].split('-')[0], \
                            config['models']['monotemp_model']['arch'].split('-')[1]

        try:
            self.seg_model = smp.create_model(
                arch=decoder,
                encoder_name=encoder,
                classes=classes,
                in_channels=channels,
                img_size=img_size,
            )
        except (KeyError, TypeError):
            # Try with 'tu-' prefix, and possibly without img_size
            try:
                self.seg_model = smp.create_model(
                    arch=decoder,
                    encoder_name='tu-' + encoder,
                    classes=classes,
                    in_channels=channels,
                    img_size=img_size,
                )
            except TypeError:
                # Fallback: no img_size
                self.seg_model = smp.create_model(
                    arch=decoder,
                    encoder_name='tu-' + encoder,
                    classes=classes,
                    in_channels=channels,
                )
        
        # Read the gradient checkpointing config flag
        use_checkpoint = config.get('models', {}).get('use_gradient_checkpointing', False)
        
        if self.return_type == 'encoder':
            self.seg_model = self.seg_model.encoder
        elif self.return_type == 'decoder':
            self.seg_model = DecoderWrapper(self.seg_model.decoder, self.seg_model.segmentation_head, use_checkpoint=use_checkpoint)