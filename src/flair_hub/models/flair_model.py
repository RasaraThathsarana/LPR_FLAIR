import numpy as np
import math
import random
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from flair_hub.models.monotemp_model import FLAIR_Monotemp
from flair_hub.models.multitemp_model import UTAE
from flair_hub.models.lpr_adapter import LPRAdapter
from flair_hub.models.refiner_modules import LocalPatchRefiner


class FLAIR_HUB_Model(nn.Module):
    """
    FLAIR_HUB_Model is a multi-modal deep learning model for satellite image processing.
    """

    def __init__(self, config: dict, img_input_sizes: dict):
        super().__init__()
        self.config = config
        self.img_input_sizes = img_input_sizes
        
        self.mono_keys = ['AERIAL_RGBI', 'AERIAL-RLT_PAN', 'DEM_ELEV', 'SPOT_RGBI']
        self.multi_keys = ['SENTINEL2_TS', 'SENTINEL1-ASC_TS', 'SENTINEL1-DESC_TS']
        
        # Auxiliary losses
        self.aux_losses = {
            mod: loss
            for mod, loss in config['modalities']['aux_loss'].items()
            if loss and config['modalities']['inputs'].get(mod, False)
        }
        
        # Tasks
        self.tasks = len(config['labels'])
        self.task_nclasses = sum(
            len(config['labels_configs'][label]['value_name'])
            for label in config['labels']
        )
        
        # Gather modality channels
        self.channels_dict = {
            mod: (
                1 if mod in ['AERIAL-RLT_PAN', 'DEM_ELEV']
                else (len(config['modalities']['inputs_channels'][mod])
                      if mod in config['modalities']['inputs_channels']
                      else 0)
            )
            for mod in config['modalities']['inputs']
        }
        
        if config['modalities']['inputs']['DEM_ELEV']:
            self.channels_dict['DEM_ELEV'] = (
                1 if config['modalities']['pre_processings']['calc_elevation']
                and not config['modalities']['pre_processings']['calc_elevation_stack_dsm']
                else 2
            )
        
        # Encoders
        self.encoders = nn.ModuleDict()
        for mod in self.mono_keys:
            if config['modalities']['inputs'].get(mod, False):

                encoder = FLAIR_Monotemp(
                    config,
                    channels=self.channels_dict[mod],
                    classes=self.task_nclasses,
                    img_size=img_input_sizes[mod],
                    return_type='encoder',
                )
                if config.get('models', {}).get('use_gradient_checkpointing', False):
                    # ENABLE GRADIENT CHECKPOINTING HERE
                    if hasattr(encoder.seg_model, "model"):
                        encoder.seg_model.model.set_grad_checkpointing(True)

                self.encoders[mod] = encoder
        
        # Adjust UTAE parameters for multi-modal fusion
        if any(config['modalities']['inputs'].get(key, False) for key in self.multi_keys):
            if self.task_nclasses != config['models']['multitemp_model']['out_conv'][-1]:
                config['models']['multitemp_model']['out_conv'].append(self.task_nclasses)

        if self.encoders:
            mono_encoder_out_channels = next(iter(self.encoders.values())).seg_model.out_channels
            config['models']['multitemp_model']['encoder_widths'] = self.adjust_fm_length(
                config, mono_encoder_out_channels
            )
            config['models']['multitemp_model']['decoder_widths'] = self.adjust_fm_length(
                config, mono_encoder_out_channels
            )
        
        for mod in self.multi_keys:
            if config['modalities']['inputs'].get(mod, False):
                self.encoders[mod] = UTAE(
                    input_dim=len(config['modalities']['inputs_channels'][mod]),
                    encoder_widths=config['models']['multitemp_model']['encoder_widths'],
                    decoder_widths=config['models']['multitemp_model']['decoder_widths'],
                    out_conv=config['models']['multitemp_model']['out_conv'],
                    str_conv_k=config['models']['multitemp_model']['str_conv_k'],
                    str_conv_s=config['models']['multitemp_model']['str_conv_s'],
                    str_conv_p=config['models']['multitemp_model']['str_conv_p'],
                    agg_mode=config['models']['multitemp_model']['agg_mode'],
                    encoder_norm=config['models']['multitemp_model']['encoder_norm'],
                    n_head=config['models']['multitemp_model']['n_head'],
                    d_model=config['models']['multitemp_model']['d_model'],
                    d_k=config['models']['multitemp_model']['d_k'],
                    encoder=False,
                    return_maps=True,
                    pad_value=config['models']['multitemp_model']['pad_value'],
                    padding_mode=config['models']['multitemp_model']['padding_mode'],
                )
        
        if any(mod in self.encoders for mod in self.mono_keys):
            encoders_out_channels = self.calc_backbones_channels()
            print(f"Calculated backbone output channels for fusion: {encoders_out_channels}")
            target_fused_channels = next(iter(self.encoders.values())).seg_model.out_channels
        else:
            encoders_out_channels = [1]  # Dummy value
            target_fused_channels = [1]  # Dummy value
        
        use_checkpoint = config.get('models', {}).get('use_gradient_checkpointing', False)

        self.fusion_handler = FusionHandler(
            backbones_channels=encoders_out_channels,
            target_fused_channels=target_fused_channels,
            mono_keys=self.mono_keys,
            multi_keys=self.multi_keys,
            use_checkpoint=use_checkpoint
        )                  
        
        self.use_LPR_decoder = config['models']['monotemp_model'].get('use_LPR_decoder', False)

        # Main Decoders
        self.main_decoders = nn.ModuleDict()
        
        # Memory Optimization: Don't initialize unused standard decoders if LPR is active
        if not self.use_LPR_decoder:
            for task in config['labels']:
                self.main_decoders[task] = (
                    FLAIR_Monotemp(
                        config,
                        channels=1,
                        classes=len(config['labels_configs'][task]['value_name']),
                        return_type='decoder',
                    )
                    if any(mod in self.encoders for mod in self.mono_keys)
                    else nn.Conv2d(
                        in_channels=self.task_nclasses,
                        out_channels=len(config['labels_configs'][task]['value_name']),
                        kernel_size=1,
                    )
                )
        
        # Auxiliary Decoders
        self.aux_decoders = nn.ModuleDict()
        for task in config['labels']:
            for mod in self.aux_losses.keys():
                self.aux_decoders[f'{mod}__{task}'] = (
                    FLAIR_Monotemp(
                        config,
                        channels=1,
                        classes=len(config['labels_configs'][task]['value_name']),
                        return_type='decoder',
                    )
                    if mod in self.mono_keys
                    else nn.Conv2d(
                        in_channels=self.task_nclasses,
                        out_channels=len(config['labels_configs'][task]['value_name']),
                        kernel_size=1,
                    )
                )

        # LPR Adapter & Refiner Setup
        if self.use_LPR_decoder:
            self.lpr_adapter = LPRAdapter(use_checkpoint=use_checkpoint)
            self.refiner = LocalPatchRefiner(
                global_dim=512, in_channels=3, patch_size=16, hidden_dim=64, cnn_dim=32,
                use_checkpoint=use_checkpoint
            )
            self.refiner_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Conv2d(32, self.task_nclasses, 1)
            )
            
            # LPR Auxiliary Decoder for Patch-wise Multi-Class Classification
            self.use_lpr_aux_decoder = config.get('models', {}).get('monotemp_model', {}).get('use_lpr_aux_decoder', False)
            if self.use_lpr_aux_decoder:
                self.lpr_aux_decoders = nn.ModuleDict()
                for task in config['labels']:
                    task_nclasses = len(config['labels_configs'][task]['value_name'])
                    self.lpr_aux_decoders[task] = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.LayerNorm(128),
                        nn.ReLU(),
                        nn.Linear(128, task_nclasses)
                    )
                print("Local Patch Refiner Auxiliary Decoder added for patch-wise classification.")
            else:
                self.lpr_aux_decoders = None
            print("Local Patch Refiner Architecture added to the model.")
        else:
            self.lpr_adapter = None
            self.refiner = None
            self.refiner_head = None
            self.use_lpr_aux_decoder = False
            self.lpr_aux_decoders = None

        self.print_model_parameters(
            self.encoders, self.main_decoders, self.aux_decoders,
            self.mono_keys, self.multi_keys, self.config
        )

        train_backbone = self.config.get('tasks', {}).get('train_tasks', {}).get('train_backbone', True)
        if not train_backbone:
            for param in self.encoders.parameters():
                param.requires_grad = False

    def adjust_fm_length(self, config: dict, mono_temp_backbone_channels: list) -> list:
        def round_to_nearest_power_of_eight(x):
            return 2 ** round(math.log(x, 2))

        if len(mono_temp_backbone_channels) > 2 and (mono_temp_backbone_channels[0] == 0 or mono_temp_backbone_channels[1] == 0):
            mono_temp_backbone_channels = mono_temp_backbone_channels[2:]
            
        encoder_widths = config['models']['multitemp_model']['encoder_widths']
        min_value = min(encoder_widths)
        max_value = max(encoder_widths)
        
        target_size = len(mono_temp_backbone_channels)
        expanded_widths = np.linspace(min_value - 1, max_value + 1, target_size).astype(int)
        
        return [round_to_nearest_power_of_eight(value) for value in expanded_widths]

    @rank_zero_only
    def print_model_parameters(
        self, 
        encoders: nn.ModuleDict, 
        decoders_main: nn.ModuleDict, 
        decoders_aux: nn.ModuleDict, 
        mono_keys: list[str], 
        multi_keys: list[str], 
        config: dict
    ) -> None:
        total_params = 0
        table = " " + "-"*113 + "\n"
        table += ("| {:<37} | {:<35} | {:<17} | {:<13} |\n"
                "| {} | {} | {} | {} |\n").format("Model modality", "Architecture", "Type", "Parameters",
                                                    "-"*37, "-"*35, "-"*17, "-"*13)

        has_mono_key = any(key in mono_keys for key in encoders)
        default_decoder_arch = 'utae'
        
        for key, model in encoders.items():
            if model is not None:
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                if key in mono_keys:
                    encoder = config['models']['monotemp_model']['arch'].split('-')[0]
                elif key in multi_keys:
                    encoder = 'utae'
                else:
                    encoder = 'Unknown'
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, encoder, 'backbone', num_params)

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*35, "-"*17, "-"*13)
        for key, model in decoders_aux.items():
            if model is not None:
                first_part_of_key = key.split('__')[0]
                if first_part_of_key in multi_keys:
                    decoder_arch = 'utae'
                else:
                    decoder_arch = config['models']['monotemp_model']['arch'].split('-')[1] if has_mono_key else 'utae'
                num_params = sum(p.numel() for p in model.parameters())
                total_params += num_params
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'aux loss decoder', num_params)

        table += "| {} | {} | {} | {} |\n".format("-"*37, "-"*35, "-"*17, "-"*13)
        
        if hasattr(self, 'use_LPR_decoder') and self.use_LPR_decoder:
            lpr_params = sum(p.numel() for p in self.lpr_adapter.parameters()) + \
                         sum(p.numel() for p in self.refiner.parameters()) + \
                         sum(p.numel() for p in self.refiner_head.parameters())
            total_params += lpr_params
            table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format("LPR Decoder", "LocalPatchRefiner", "refiner_head", lpr_params)
            if getattr(self, 'use_lpr_aux_decoder', False) and self.lpr_aux_decoders is not None:
                lpr_aux_params = sum(p.numel() for p in self.lpr_aux_decoders.parameters())
                total_params += lpr_aux_params
                table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format("LPR Aux Decoder", "Sequential", "lpr_aux_decoders", lpr_aux_params)
        else:
            for key, model in decoders_main.items():
                if model is not None:
                    if has_mono_key:
                        decoder_arch = config['models']['monotemp_model']['arch'].split('-')[1]
                    else:
                        decoder_arch = default_decoder_arch
                    num_params = sum(p.numel() for p in model.parameters())
                    total_params += num_params
                    table += "| {:<37} | {:<35} | {:<17} | {:>13,} |\n".format(key, decoder_arch, 'task decoder', num_params)

        # Total Parameters
        table += "|" + "-"*113 +'|'
        table += "\n| {:<37}   {:<35}   {:<17}   {:>13,} |\n".format("Total parameters", "", "", total_params)
        table += ' '+"-"*113

        print('')
        print(table)

    def calc_backbones_channels(self) -> list[int]:
        backbones_total_channels = []
        for mod in self.encoders:
            if mod in self.mono_keys:
                out_channels = self.encoders[mod].seg_model.out_channels
                if len(out_channels) > 2 and (out_channels[0] == 0 or out_channels[1] == 0):
                    channels = out_channels[2:]
                else:
                    channels = out_channels  
                backbones_total_channels.append(list(channels))

        reversed_decoder_fm = list(self.config['models']['multitemp_model']['decoder_widths'])[::-1]
        for mod in self.encoders:
            if mod in self.multi_keys:
                backbones_total_channels.append(reversed_decoder_fm)
        total_per_stage = [sum(x) for x in zip(*backbones_total_channels)]

        return total_per_stage

    def interpolate_map(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def modality_dropout(
        self, 
        feature_maps: dict[str, list[torch.Tensor]], 
        modalities_dropout_dict: dict[str, float]
    ) -> dict[str, list[torch.Tensor]]:
        for key in feature_maps.keys():
            dropout_prob = modalities_dropout_dict[key]
            if torch.rand(1).item() < dropout_prob:
                param_features = []
                for tensor in feature_maps[key]:     
                    param_tensor = nn.Parameter(torch.empty_like(tensor))
                    nn.init.xavier_uniform_(param_tensor)
                    param_features.append(param_tensor)
                feature_maps[key] = param_features
        return feature_maps
    
    def forward(self, batch: dict, apply_mod_dropout: bool = False) -> dict:
        fmaps, logits_tasks, logits_aux = {}, {}, {}

        active_mono_keys = [key for key in self.mono_keys if key in self.encoders]
        active_multi_keys = [key for key in self.multi_keys if key in self.encoders]
        
        # Bug fix: use shape[-2:] (H,W) to ensure correct interpolation instead of trailing dim
        img_size = batch[self.config['labels'][0]].shape[-2:]

        for mod, encoder in self.encoders.items():
            if mod in self.mono_keys:
                fmaps[mod] = encoder.seg_model(batch[mod])

                if self.aux_losses.get(mod):
                    for task in self.config['labels']:
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            logits_aux[f'aux_{mod}_{task}'] = self.interpolate_map(
                                aux_decoder.seg_model(*fmaps[mod]), img_size
                            )
            else:
                logits, fmaps_ = encoder(batch[mod], batch_positions=batch.get(mod.replace('TS', 'DATES')))
                logits = self.interpolate_map(logits, img_size)

                logits_tasks[mod] = self.interpolate_map(logits, img_size)
                fmaps[mod] = fmaps_

                if self.aux_losses.get(mod):
                    for task in self.config['labels']:
                        aux_key = f"{mod}__{task}"
                        if aux_key in self.aux_decoders:
                            aux_decoder = self.aux_decoders[aux_key]
                            logits_aux[f'aux_{mod}_{task}'] = self.interpolate_map(
                                aux_decoder(logits), img_size
                            )

        if apply_mod_dropout and len(self.encoders) > 1:
            modalities_dropout_dict = {key: random.uniform(0, 1) for key in fmaps.keys()}
            fmaps = self.modality_dropout(fmaps, modalities_dropout_dict)

        if active_mono_keys:
            fused_features = self.fusion_handler(fmaps, fmaps[active_mono_keys[0]])
        else:
            fused_features = self.fusion_handler(logits_tasks, logits_tasks[active_multi_keys[0]])

        if self.use_LPR_decoder:
            img, global_tokens = self.lpr_adapter(fused_features)
            lpr_refined_features = self.refiner(img, global_tokens)
            
            use_checkpoint = self.config.get('models', {}).get('use_gradient_checkpointing', False)
            if use_checkpoint and lpr_refined_features.requires_grad:
                lpr_out = checkpoint(self.refiner_head, lpr_refined_features, use_reentrant=False)
            else:
                lpr_out = self.refiner_head(lpr_refined_features)

            # Properly map refiner output across multiple tasks
            start_idx = 0
            for task in self.config['labels']:
                n_classes = len(self.config['labels_configs'][task]['value_name'])
                logits_tasks[task] = self.interpolate_map(lpr_out[:, start_idx:start_idx+n_classes, ...], img_size)
                start_idx += n_classes
                
            if getattr(self, 'use_lpr_aux_decoder', False) and self.lpr_aux_decoders is not None:
                for task in self.config['labels']:
                    gt = global_tokens
                    if gt.ndim == 4:
                        gt = gt.flatten(2).transpose(1, 2) # (B, N, C)
                        
                    if use_checkpoint and gt.requires_grad:
                        aux_out = checkpoint(self.lpr_aux_decoders[task], gt, use_reentrant=False)
                    else:
                        aux_out = self.lpr_aux_decoders[task](gt)
                        
                    logits_aux[f'lpr_aux_{task}'] = aux_out.transpose(1, 2) # (B, task_nclasses, N)
        else:   
            for task in self.config['labels']:
                if active_mono_keys:
                    logits_tasks[task] = self.interpolate_map(
                        self.main_decoders[task].seg_model(*fused_features), img_size
                    )
                else:
                    if len(self.config['labels']) > 1:
                        logits_tasks[task] = self.main_decoders[task](fused_features)
                    else:
                        logits_tasks[task] = fused_features

            for mod in list(logits_tasks.keys()):
                if mod in self.config['modalities']['inputs']:
                    del logits_tasks[mod]

        return logits_tasks, logits_aux

class FusionHandler(nn.Module):
    def __init__(
        self,
        backbones_channels: list[int],
        target_fused_channels: list[int],
        mono_keys: list[str],
        multi_keys: list[str],
        use_checkpoint: bool = False
    ) -> None:
        super().__init__()

        self.mono_keys = mono_keys
        self.multi_keys = multi_keys
        self.use_checkpoint = use_checkpoint

        if len(target_fused_channels) > 2 and (
            target_fused_channels[0] == 0 or target_fused_channels[1] == 0
        ):
            target_fused_channels = target_fused_channels[2:]

        self.conv_f = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(backbones_channels, target_fused_channels)
            ]
        )

    def _forward_impl(self, feature_maps: dict, target_fm_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        active_keys = list(feature_maps.keys())
        mono_key_active = [key for key in active_keys if key in self.mono_keys]
        multi_key_active = [key for key in active_keys if key in self.multi_keys]

        if len(mono_key_active) == 1 and len(multi_key_active) == 0:
            return feature_maps[mono_key_active[0]]

        if len(mono_key_active) == 0 and len(multi_key_active) == 1:
            return feature_maps[multi_key_active[0]]

        if len(mono_key_active) == 0 and len(multi_key_active) > 1:
            stacked_feature_maps = torch.stack(
                [feature_maps[key] for key in multi_key_active], dim=0
            )
            return torch.mean(stacked_feature_maps, dim=0)

        target_shapes = [fm.shape for fm in target_fm_maps]

        if target_shapes[0][1] == 0 or target_shapes[1][1] == 0:
            target_shapes = target_shapes[2:]
            dummy_shapes = target_fm_maps[:2]
        else:
            dummy_shapes = None

        aligned_fmaps = []

        for mod in active_keys:
            mod_fmaps = feature_maps[mod]

            if mod_fmaps[0].shape[1] == 0 or mod_fmaps[1].shape[1] == 0:
                mod_fmaps = mod_fmaps[2:]

            if len(mod_fmaps) != len(target_shapes):
                mod_fmaps = [mod_fmaps[0]] * (len(target_shapes) - len(mod_fmaps)) + mod_fmaps

            resized_fmaps = []
            for fmap, target in zip(mod_fmaps, target_shapes):
                target_h, target_w = target[-2], target[-1]

                if fmap.shape[-1] != target_w or fmap.shape[-2] != target_h:
                    fmap = F.interpolate(fmap, size=(target_h, target_w), mode='bilinear', align_corners=False)

                resized_fmaps.append(fmap)

            aligned_fmaps.append(resized_fmaps)

        stacked_feature_maps = [torch.cat(fmaps, dim=1) for fmaps in zip(*aligned_fmaps)]

        adjusted_feature_maps = [
            conv_f(fmap) for conv_f, fmap in zip(self.conv_f, stacked_feature_maps)
        ]

        if dummy_shapes is not None:
            adjusted_feature_maps = dummy_shapes + adjusted_feature_maps

        return adjusted_feature_maps

    def forward(self, feature_maps: dict, target_fm_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        # Check if any input tensor requires gradients
        requires_grad = False
        for fmaps in feature_maps.values():
            if any(isinstance(t, torch.Tensor) and t.requires_grad for t in fmaps):
                requires_grad = True
                break
        
        if self.use_checkpoint and requires_grad:
            return checkpoint(self._forward_impl, feature_maps, target_fm_maps, use_reentrant=False)
        else:
            return self._forward_impl(feature_maps, target_fm_maps)