import os
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, BasePredictionWriter
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision, MultilabelRecall

# FLAIR-HUB imports
from flair_hub.tasks.module_setup import build_data_module, get_input_img_sizes
from flair_hub.models.flair_model import FLAIR_HUB_Model

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None


def load_configs(config_folder: str) -> dict:
    """Helper to load all YAML config files from a directory into a single dictionary."""
    config = {}
    for filename in os.listdir(config_folder):
        if filename.endswith(('.yaml', '.yml')):
            with open(os.path.join(config_folder, filename), 'r') as f:
                parsed = yaml.safe_load(f)
                if parsed:
                    config.update(parsed)
    return config


def load_encoder_weights(model: nn.Module, ckpt_path: str):
    """Loads encoder weights from a segmentation checkpoint into the model."""
    print(f"Loading encoder weights from {ckpt_path}...")
    if ckpt_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Please install safetensors to load .safetensors files")
        state_dict = safe_load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
    # Strip 'model.' prefix if the model was saved as a PyTorch Lightning module
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("model."):] if k.startswith("model.") else k
        new_state_dict[new_key] = v
        
    # Load the state dict. strict=False ensures we ignore missing decoder weights.
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if len(missing) > 0:
        print(f"Successfully loaded weights. Note: {len(missing)} keys were missing (as expected, since some layers like decoders might be ignored).")
    else:
        print("Successfully loaded weights. All expected keys from the checkpoint were loaded.")
    

class MultiLabelClassificationTask(pl.LightningModule):
    def __init__(self, config: dict, in_img_sizes: dict, task_name: str):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.num_classes = len(config['labels_configs'][task_name]['value_name'])
        
        # Initialize the base segmentation model (only the encoders and fusion will be utilized)
        self.flair_model = FLAIR_HUB_Model(config, in_img_sizes)
        
        # Attempt to dynamically determine the output channels of the encoder/fusion backbone
        cls_in_channels = 512  # Fallback guess
        try:
            active_mono_keys = [k for k in self.flair_model.mono_keys if k in self.flair_model.encoders]
            if active_mono_keys:
                out_channels = self.flair_model.encoders[active_mono_keys[0]].seg_model.out_channels
                if isinstance(out_channels, (list, tuple)) and len(out_channels) >= 2:
                    cls_in_channels = out_channels[-1] + out_channels[-2]
                else:
                    cls_in_channels = out_channels[-1] if isinstance(out_channels, (list, tuple)) else out_channels
            elif hasattr(self.flair_model, 'fusion_handler') and len(self.flair_model.fusion_handler.conv_f) >= 2:
                cls_in_channels = self.flair_model.fusion_handler.conv_f[-1].out_channels + self.flair_model.fusion_handler.conv_f[-2].out_channels
            elif hasattr(self.flair_model, 'fusion_handler') and len(self.flair_model.fusion_handler.conv_f) > 0:
                cls_in_channels = self.flair_model.fusion_handler.conv_f[-1].out_channels
        except Exception as e:
            print(f"Warning: Could not automatically infer encoder output channels. Defaulting to {cls_in_channels}. Error: {e}")
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Simple but effective Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(cls_in_channels, self.num_classes)
        )
        
        # BCE With Logits is the standard for multi-label classification
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.val_acc = MultilabelAccuracy(num_labels=self.num_classes)
        
        # Overall (macro) validation metrics
        self.val_f1 = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.val_precision = MultilabelPrecision(num_labels=self.num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_labels=self.num_classes, average="macro")

        # Per-class validation metrics
        self.val_f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average=None)
        self.val_precision_per_class = MultilabelPrecision(num_labels=self.num_classes, average=None)
        self.val_recall_per_class = MultilabelRecall(num_labels=self.num_classes, average=None)

        # Freeze backbone if specified in config
        train_backbone = self.config.get('tasks', {}).get('train_tasks', {}).get('train_backbone', True)
        self.flair_model.requires_grad_(train_backbone) # Cleaner, idiomatic PyTorch way to freeze
        if not train_backbone:
            print("\n[!] train_backbone is False. Freezing backbone weights. Only the classification head will be trained.\n")
        else:
            print("\n[ ] train_backbone is True. The full model is trainable.\n")

    def forward(self, batch):
        # Replicate the encoder forward path from FLAIR_HUB_Model to avoid running the heavy decoders
        fmaps = {}
        for mod, encoder in self.flair_model.encoders.items():
            if mod in self.flair_model.mono_keys:
                fmaps[mod] = encoder.seg_model(batch[mod])
            else:
                _, fmaps_ = encoder(batch[mod], batch_positions=batch.get(mod.replace('TS', 'DATES')))
                fmaps[mod] = fmaps_
                
        active_mono_keys = [key for key in self.flair_model.mono_keys if key in self.flair_model.encoders]
        active_multi_keys = [key for key in self.flair_model.multi_keys if key in self.flair_model.encoders]
        
        # Perform fusion to get the combined feature maps
        if active_mono_keys:
            fused_features = self.flair_model.fusion_handler(fmaps, fmaps[active_mono_keys[0]])
        elif active_multi_keys:
            fused_features = self.flair_model.fusion_handler(fmaps, fmaps[active_multi_keys[0]])
        else:
            raise ValueError("No active modalities found in the batch.")
            
        # Extract features and apply spatial pooling
        if isinstance(fused_features, (list, tuple)):
            if len(fused_features) >= 2:
                feat1 = self.pool(fused_features[-1]).flatten(1)
                feat2 = self.pool(fused_features[-2]).flatten(1)
                features = torch.cat([feat1, feat2], dim=1)
            else:
                features = self.pool(fused_features[-1]).flatten(1)
        else:
            features = self.pool(fused_features).flatten(1)

        # Classification Head
        logits = self.classifier(features)
        return logits

    def _get_multilabel_targets(self, batch):
        """Transforms spatial segmentation masks into multi-label classification targets dynamically."""
        targets = batch[self.task_name].to(self.device)
        if targets.ndim == 4:
            targets = torch.argmax(targets, dim=1) # Shape becomes [B, H, W]
            
        B = targets.size(0)
        multi_targets = torch.zeros((B, self.num_classes), device=self.device, dtype=torch.float32)
        
        targets_flat = targets.view(B, -1) # Flatten spatial dimensions
        for b in range(B):
            unique_classes = torch.unique(targets_flat[b])
            # Filter valid class boundaries (avoid ignoring indices like 255)
            valid_classes = unique_classes[(unique_classes >= 0) & (unique_classes < self.num_classes)]
            multi_targets[b, valid_classes.long()] = 1.0
            
        return multi_targets

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        targets = self._get_multilabel_targets(batch)
        loss = self.loss_fn(logits, targets)
        
        self.train_acc(logits, targets.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Logs learning rate at the end of a training epoch."""
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = self._get_multilabel_targets(batch)
        loss = self.loss_fn(logits, targets)
        
        self.val_acc(logits, targets.long())
        self.val_f1(logits, targets.long())
        self.val_precision(logits, targets.long())
        self.val_recall(logits, targets.long())
        self.val_f1_per_class(logits, targets.long())
        self.val_precision_per_class(logits, targets.long())
        self.val_recall_per_class(logits, targets.long())
        
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Logs per-class metrics at the end of the validation epoch."""
        class_names = self.config['labels_configs'][self.task_name]['value_name']

        # F1 per class
        f1_per_class = self.val_f1_per_class.compute()
        for i, f1 in enumerate(f1_per_class):
            class_name = class_names.get(i, f"class_{i}")
            self.log(f"val_f1_class_{i}_{class_name}", f1, sync_dist=True)
        self.val_f1_per_class.reset()

        # Precision per class
        precision_per_class = self.val_precision_per_class.compute()
        for i, precision in enumerate(precision_per_class):
            class_name = class_names.get(i, f"class_{i}")
            self.log(f"val_precision_class_{i}_{class_name}", precision, sync_dist=True)
        self.val_precision_per_class.reset()

        # Recall per class
        recall_per_class = self.val_recall_per_class.compute()
        for i, recall in enumerate(recall_per_class):
            class_name = class_names.get(i, f"class_{i}")
            self.log(f"val_recall_class_{i}_{class_name}", recall, sync_dist=True)
        self.val_recall_per_class.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        output = {
            "preds": preds,
            "probs": probs,
            "ids": batch.get(f"ID_{self.task_name}", [f"batch_{batch_idx}_{i}" for i in range(preds.size(0))])
        }
        
        if self.task_name in batch:
            output["targets"] = self._get_multilabel_targets(batch)
        return output

    def configure_optimizers(self):
        cfg = self.config['hyperparams']
        lr = cfg.get('learning_rate', 1e-4)
        params_to_train = filter(lambda p: p.requires_grad, self.parameters())
        
        # Setup optimizer from config
        optim_type = cfg.get('optimizer', 'adamw')
        if optim_type == 'sgd':
            optimizer = torch.optim.SGD(params_to_train, lr=lr)
        elif optim_type in ['adam', 'adamw']:
            OptimClass = torch.optim.AdamW if optim_type == 'adamw' else torch.optim.Adam
            optimizer = OptimClass(
                params_to_train, 
                lr=lr, 
                weight_decay=cfg.get('optim_weight_decay', 0.01), 
                betas=tuple(cfg.get('optim_betas', [0.9, 0.999]))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")

        # Check for scheduler in config
        scheduler_type = cfg.get("scheduler", None)
        if not scheduler_type:
            return optimizer

        if scheduler_type == "one_cycle_lr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=cfg.get("warmup_fraction", 0.2),
                cycle_momentum=False,
                div_factor=1000
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        
        elif scheduler_type == "cosine_annealing_lr":
            total_steps = cfg.get('total_steps', self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=cfg.get('eta_min', 1e-5) # Strictly defaults to 0.00001 if omitted in config
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min",
                factor=0.5, 
                patience=cfg.get('plateau_patience', 5),
                min_lr=1e-7
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}}

        print(f"Warning: Scheduler '{scheduler_type}' is defined in config but not implemented in MultiLabelClassificationTask. Using fixed LR.")
        return optimizer


class ClassificationPredictionWriter(BasePredictionWriter):
    """Custom writer to save multi-label predictions and probabilities directly to a CSV."""
    def __init__(self, output_dir: str, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        all_preds, all_probs, all_ids, all_targets = [], [], [], []
        
        # Aggregate all results from batches
        # Handle both single dataloader (list of dicts) and multiple dataloaders (list of lists)
        pred_list = predictions[0] if (predictions and isinstance(predictions[0], list)) else predictions
        for batch_pred in pred_list:
            all_preds.append(batch_pred["preds"].cpu().numpy())
            all_probs.append(batch_pred["probs"].cpu().numpy())
            all_ids.extend(batch_pred["ids"])
            if "targets" in batch_pred:
                all_targets.append(batch_pred["targets"].cpu().numpy())
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build and write CSV
        class_names = [f"class_{i}" for i in range(all_preds.shape[1])]
        # Get actual class names from config if possible
        labels_config = getattr(pl_module, 'config', {}).get('labels_configs', {}).get(pl_module.task_name, {}).get('value_name', {})
        class_names = [labels_config.get(i, f"class_{i}") for i in range(all_preds.shape[1])]
        
        df_preds = pd.DataFrame(all_preds, columns=class_names)
        df_preds["ID"] = [x[0] if isinstance(x, (list, tuple)) else x for x in all_ids]
        df_preds.to_csv(os.path.join(self.output_dir, "classification_predictions.csv"), index=False)
        
        df_probs = pd.DataFrame(all_probs, columns=class_names)
        df_probs["ID"] = df_preds["ID"]
        df_probs.to_csv(os.path.join(self.output_dir, "classification_probabilities.csv"), index=False)
        
        # Save metrics if targets are available
        if all_targets is not None:
            try:
                from sklearn.metrics import classification_report
                
                # 1. Save standard text report
                report_txt = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
                with open(os.path.join(self.output_dir, "classification_metrics.txt"), "w") as f:
                    f.write(report_txt)
                
                # 2. Save comprehensive CSV metrics
                report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
                pd.DataFrame(report_dict).transpose().to_csv(os.path.join(self.output_dir, "classification_metrics.csv"))
                print(f"\n[✓] Metrics saved to {os.path.join(self.output_dir, 'classification_metrics.csv')}")
            except ImportError:
                print("\n[!] scikit-learn is not installed. Could not generate metrics report. Run 'pip install scikit-learn' to enable this.")
        
        print(f"\n[✓] Inference complete! Results saved in {self.output_dir}")


if __name__ == "__main__":
    # Set precision for Tensor Cores for performance improvement on compatible GPUs
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="FLAIR-HUB Multi-Label Classification Wrapper")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the 4 YAML configuration files")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a model checkpoint. For 'train', this is a segmentation model to extract encoder weights from. For 'predict', this is the trained classification model.")
    parser.add_argument("--task_name", type=str, default="AERIAL_LABEL-COSIA", help="Target task name defined in config_supervision.yaml")
    parser.add_argument("--stage", type=str, choices=["train", "predict"], default="train", help="Execution mode (train or predict)")
    parser.add_argument("--out_dir", type=str, default="./classification_output", help="Directory where model checkpoints or predictions will be saved")
    
    args = parser.parse_args()
    
    # 1. Load Configurations
    config = load_configs(args.config_dir)
    
    # 2. Build DataModule directly using existing repository structure
    stage = "fit" if args.stage == "train" else "predict"
    
    dict_train, dict_val, dict_test = None, None, None
    if args.stage == "train":
        train_csv = config.get('paths', {}).get('train_csv')
        val_csv = config.get('paths', {}).get('val_csv')
        if train_csv and os.path.exists(train_csv):
            dict_train = pd.read_csv(train_csv).to_dict('list')
        if val_csv and os.path.exists(val_csv):
            dict_val = pd.read_csv(val_csv).to_dict('list')
    else:
        test_csv = config.get('paths', {}).get('test_csv')
        if test_csv and os.path.exists(test_csv):
            dict_test = pd.read_csv(test_csv).to_dict('list')
            
    data_module = build_data_module(config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test)
    in_img_sizes = get_input_img_sizes(config, data_module, stage=stage)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 5. Build Trainer and Execute
    if args.stage == "train":
        # Pre-calculate total steps to decouple scheduler from trainer state
        # This prevents the subtle VRAM increase during PyTorch Lightning's init phase
        print("[ ] Pre-calculating total training steps for scheduler...")
        num_epochs = config['hyperparams'].get('num_epochs', 10)
        config['hyperparams']['total_steps'] = len(data_module.train_dataloader()) * num_epochs

        model = MultiLabelClassificationTask(config, in_img_sizes, args.task_name)
        
        # Load pre-trained encoder weights for fine-tuning
        if os.path.isfile(args.ckpt_path):
            load_encoder_weights(model.flair_model, args.ckpt_path)
        else:
            print(f"[!] Warning: Checkpoint path {args.ckpt_path} not found. Starting with random weights.")
            
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, "checkpoints"),
            filename="cls-ckpt-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,
            monitor="val_loss",
            mode="min"
        )
        
        trainer = pl.Trainer(
            max_epochs=config['hyperparams'].get('num_epochs', 10),
            accelerator=config['hardware'].get('accelerator', 'gpu'),
            devices=config['hardware'].get('gpus_per_node', 1),
            strategy=config['hardware'].get('strategy', 'auto'),
            callbacks=[checkpoint_cb],
            precision=config['hardware'].get("precision", "32-true")
        )
        print("\n[ ] Starting Classification Training...\n")
        trainer.fit(model, datamodule=data_module)
        
    elif args.stage == "predict":
        # For prediction, load the entire trained classification model from its checkpoint
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"Prediction checkpoint not found at: {args.ckpt_path}")

        print(f"\n[ ] Loading classification model from checkpoint: {args.ckpt_path}\n")
        model = MultiLabelClassificationTask.load_from_checkpoint(
            args.ckpt_path,
            config=config,
            in_img_sizes=in_img_sizes,
            task_name=args.task_name,
            strict=False  # Allows loading even if decoder weights are missing in the checkpoint
        )
        
        writer_cb = ClassificationPredictionWriter(output_dir=args.out_dir)
        
        trainer = pl.Trainer(
            accelerator=config['hardware'].get('accelerator', 'gpu'),
            devices=config['hardware'].get('gpus_per_node', 1),
            strategy=config['hardware'].get('strategy', 'auto'),
            callbacks=[writer_cb],
            logger=False,
            precision=config['hardware'].get("precision", "32-true")
        )
        print("\n[ ] Starting Classification Inference...\n")
        trainer.predict(model, datamodule=data_module, return_predictions=False)
