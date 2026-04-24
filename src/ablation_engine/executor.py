#  src/ablation_engine/executor.py

import torch
import torch.nn as nn
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, Any, List, Optional

from .parser import AblationConfigParser
from .modifier import ModelModifier

from ..data.subset_creator import SubsetCreator
from ..train import train_and_evaluate
from ..models.factory import create_model


class TransferLearningManager:

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_pretrained(
            self,
            model: nn.Module,
            pretrained_path: str,
            device: torch.device,
            strict: bool = False
    ) -> nn.Module:
        """
        Load pretrained weights.

        Args:
            model: Model instance
            pretrained_path: Path to pretrained weights
            device: Device to load to
            strict: Whether to strictly match state dict keys
        """
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            self.logger.warning(f"⚠️ Pretrained weights not found: {pretrained_path}")
            return model

        self.logger.info(f"📦 Loading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location=device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

            if missing_keys:
                self.logger.info(f"   ℹ️ Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for k in missing_keys:
                        self.logger.debug(f"      - {k}")

            if unexpected_keys:
                self.logger.info(f"   ℹ️ Unexpected keys: {len(unexpected_keys)}")

            self.logger.info("   ✅ Pretrained weights loaded successfully")

        except RuntimeError as e:
            self.logger.error(f"   ❌ Failed to load weights: {e}")
            raise

        return model

    def apply_finetune_mode(
            self,
            model: nn.Module,
            finetune_mode: str,
            train_params: Dict[str, Any]
    ) -> nn.Module:
        """
        Freeze/unfreeze parameters according to the fine-tuning mode.

        Args:
            model: Model instance
            finetune_mode: Fine-tuning mode
            train_params: Training parameters
        """
        if finetune_mode is None or finetune_mode == 'full':
            self.logger.info("   🔓 Finetune mode: FULL (all parameters trainable)")
            return model

        self.logger.info(f"   🔧 Applying finetune mode: {finetune_mode}")

        if finetune_mode == 'freeze_encoder':
            model = self._freeze_encoder(model)

        elif finetune_mode == 'freeze_encoder_partial':
            freeze_stages = train_params.get('freeze_stages', 2)
            model = self._freeze_encoder_partial(model, freeze_stages)

        elif finetune_mode == 'freeze_branch':
            freeze_branches = train_params.get('freeze_branches', [])
            model = self._freeze_branches(model, freeze_branches)

        elif finetune_mode == 'head_only':
            model = self._freeze_all_except_head(model)

        elif finetune_mode == 'decoder_partial':
            trainable_stages = train_params.get('trainable_decoder_stages', 2)
            model = self._freeze_decoder_partial(model, trainable_stages)

        else:
            self.logger.warning(f"   ⚠️ Unknown finetune mode: {finetune_mode}")

        self._print_trainable_stats(model)

        return model

    def _freeze_encoder(self, model: nn.Module) -> nn.Module:
        frozen_count = 0
        decoder_patterns = ['decoder', 'head', 'norm', 'up.']

        for name, param in model.named_parameters():
            if not any(p in name for p in decoder_patterns):
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(f"      Frozen {frozen_count} encoder parameters")
        return model

    def _freeze_encoder_partial(self, model: nn.Module, freeze_stages: int) -> nn.Module:
        frozen_count = 0

        staged_prefixes = [
            'dem_stages', 's1_stages', 's2_stages',
            'dem_downsamplers', 's1_downsamplers', 's2_downsamplers',
            's1_dem_injection_fusers', 's2_dem_injection_fusers',
            's1s2_cross_fusers', 'skip_fusers', 'synergy_fusers',
        ]

        for name, param in model.named_parameters():
            should_freeze = False

            if freeze_stages > 0 and any(p in name for p in ['_embed']):
                should_freeze = True

            for i in range(freeze_stages):
                if any(f'{prefix}.{i}.' in name for prefix in staged_prefixes):
                    should_freeze = True
                    break

            if should_freeze:
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(f"      Frozen stages 0-{freeze_stages - 1}: {frozen_count} parameters")
        return model

    def _freeze_branches(self, model: nn.Module, branches: List[str]) -> nn.Module:
        frozen_count = 0

        branch_prefix_map = {
            'dem': ['dem_stages', 'dem_embed', 'dem_downsamplers',
                    's1_dem_injection_fusers', 's2_dem_injection_fusers', 'synergy_fusers'],
            's1': ['s1_stages', 's1_embed', 's1_downsamplers'],
            's2': ['s2_stages', 's2_embed', 's2_downsamplers'],
        }

        freeze_prefixes = []
        for branch in branches:
            prefixes = branch_prefix_map.get(branch.lower(), [branch])
            freeze_prefixes.extend(prefixes)

        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(f"      Frozen branches {branches}: {frozen_count} parameters")
        return model

    def _freeze_all_except_head(self, model: nn.Module) -> nn.Module:
        frozen_count = 0

        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                frozen_count += 1

        self.logger.info(f"      Frozen all except head: {frozen_count} parameters")
        return model

    def _freeze_decoder_partial(self, model: nn.Module, trainable_stages: int) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = False

        unfrozen_count = 0
        num_decoder_stages = len(model.decoder) if hasattr(model, 'decoder') else 3

        for name, param in model.named_parameters():
            if any(p in name for p in ['head', 'norm', 'up.']):
                param.requires_grad = True
                unfrozen_count += 1
                continue

            for i in range(trainable_stages):
                stage_idx = num_decoder_stages - 1 - i
                if f'decoder.{stage_idx}.' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break

        self.logger.info(f"      Unfrozen last {trainable_stages} decoder stages + head: {unfrozen_count} parameters")
        return model

    def _print_trainable_stats(self, model: nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self.logger.info(f"   📊 Parameter Statistics:")
        self.logger.info(f"      Total:     {total_params:,}")
        self.logger.info(f"      Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        self.logger.info(f"      Frozen:    {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")

        if frozen_params > 0:
            module_stats = {}
            for name, param in model.named_parameters():
                prefix = name.split('.')[0]  # Take the top-level module name
                if prefix not in module_stats:
                    module_stats[prefix] = {'total': 0, 'frozen': 0}
                module_stats[prefix]['total'] += param.numel()
                if not param.requires_grad:
                    module_stats[prefix]['frozen'] += param.numel()

            self.logger.info(f"      --- Per-module breakdown ---")
            for prefix, stats in sorted(module_stats.items()):
                if stats['frozen'] == stats['total']:
                    status = "🔒"
                elif stats['frozen'] == 0:
                    status = "🔓"
                else:
                    status = "⚡"
                self.logger.info(
                    f"      {status} {prefix:<30s}: "
                    f"{stats['frozen']:>10,} / {stats['total']:>10,} frozen"
                )

    def create_optimizer_with_layerwise_lr(
            self,
            model: nn.Module,
            train_params: Dict[str, Any]
    ) -> torch.optim.Optimizer:
        base_lr = train_params.get('lr', 0.0005)
        weight_decay = train_params.get('weight_decay', 0.05)
        finetune_mode = train_params.get('finetune_mode', 'full')

        if finetune_mode == 'layerwise_lr':
            encoder_lr_scale = train_params.get('encoder_lr_scale', 0.1)
            decoder_lr_scale = train_params.get('decoder_lr_scale', 1.0)

            encoder_params = []
            decoder_params = []

            decoder_patterns = ['decoder', 'head', 'up.', 'norm']

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(p in name for p in decoder_patterns):
                    decoder_params.append(param)
                else:
                    encoder_params.append(param)

            param_groups = []
            if encoder_params:
                param_groups.append({'params': encoder_params, 'lr': base_lr * encoder_lr_scale, 'name': 'encoder'})
            if decoder_params:
                param_groups.append({'params': decoder_params, 'lr': base_lr * decoder_lr_scale, 'name': 'decoder'})

            self.logger.info(f"   📊 Layerwise LR:")
            for g in param_groups:
                self.logger.info(f"      {g['name']}: lr={g['lr']:.6f}, params={len(g['params'])}")

        elif finetune_mode == 'layerwise_lr_gradual':
            layer_lr_scales = train_params.get('layer_lr_scales', {})
            param_groups = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                lr_scale = 1.0
                group_name = 'default'

                for layer_name, scale in layer_lr_scales.items():
                    if layer_name.lower() in name.lower():
                        lr_scale = scale
                        group_name = layer_name
                        break

                found = False
                for g in param_groups:
                    if g.get('name') == group_name:
                        g['params'].append(param)
                        found = True
                        break

                if not found:
                    param_groups.append({
                        'params': [param],
                        'lr': base_lr * lr_scale,
                        'name': group_name
                    })

            self.logger.info(f"   📊 Gradual Layerwise LR:")
            for g in param_groups:
                self.logger.info(f"      {g['name']}: lr={g['lr']:.6f}, params={len(g['params'])}")

        else:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]

        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
        return optimizer


class GradualUnfreezeScheduler:

    def __init__(
            self,
            model: nn.Module,
            schedule: List[Dict],
            logger: logging.Logger
    ):
        self.model = model
        self.schedule = schedule
        self.logger = logger
        self.current_stage = 0

    def step(self, epoch: int):
        for item in self.schedule:
            if epoch >= item['epoch'] and self.current_stage < self.schedule.index(item) + 1:
                self._unfreeze(item['unfreeze'])
                self.current_stage = self.schedule.index(item) + 1

    def _unfreeze(self, patterns: List[str]):
        unfrozen = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                should_unfreeze = False

                for pattern in patterns:
                    if pattern == 'all':
                        should_unfreeze = True
                        break
                    if pattern.lower() in name.lower():
                        should_unfreeze = True
                        break

                if should_unfreeze:
                    param.requires_grad = True
                    unfrozen += 1

        self.logger.info(f"   🔓 Unfroze {unfrozen} parameters matching {patterns}")


class AutoAblationExecutor:

    def __init__(self, config_path: str):
        self.parser = AblationConfigParser(config_path)
        self.base_config = self.parser.base_config
        self.execution_config = self.parser.execution_config

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(self.execution_config['output_dir']) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.results: List[Dict[str, Any]] = []

        self.transfer_manager = TransferLearningManager(self.logger)

    def _setup_logging(self):
        log_file = self.output_dir / 'ablation_run.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Ablation study initialized. Results will be saved to: {self.output_dir}")

    def create_model_from_config(self, exp_config: Dict[str, Any]) -> nn.Module:
        self.logger.info(f"Creating model for experiment: {exp_config['experiment_id']}")

        current_config = self.base_config
        if 'config_override' in exp_config:
            current_config = ModelModifier.apply_config_override(
                self.base_config, exp_config['config_override']
            )
            self.logger.info(f"  > Applied config override.")

        model_params = current_config.get('model_params', {})
        model = create_model(model_params)

        if 'module_replacements' in exp_config:
            model = ModelModifier.apply_module_replacement(
                model, exp_config['module_replacements'], model_params
            )

        if 'module_modifications' in exp_config:
            model = ModelModifier.apply_module_modifications(
                model, exp_config['module_modifications']
            )

        if 'forward_hook' in exp_config:
            model = ModelModifier.apply_forward_hook(
                model, exp_config['forward_hook']
            )

        return model

    def run_all(self):
        experiments_to_run = self.parser.get_experiment_configs()
        self.logger.info(f"Starting ablation study with {len(experiments_to_run)} experiments...")

        for i, exp_config_raw in enumerate(experiments_to_run, 1):
            exp_id = exp_config_raw['experiment_id']
            self.logger.info(f"\n{'=' * 25} Experiment {i}/{len(experiments_to_run)}: {exp_id} {'=' * 25}")

            try:

                exp_full_config = ModelModifier.apply_config_override(
                    self.base_config,
                    exp_config_raw.get('config_override', {})
                )

                data_params = exp_full_config.get('data_params', {})
                train_params = exp_full_config.get('training_params', {})
                model_params = exp_full_config.get('model_params', {})

                self.logger.info("Preparing datasets...")
                dataset_type = data_params.get('dataset_type', 'multi_modal')
                self.logger.info(f"  Dataset type: {dataset_type}")

                creator_kwargs = {'seed': data_params.get('seed', 42)}

                if dataset_type == 'multiyear_npz':
                    creator_kwargs.update({
                        'dataset_type': 'multiyear_npz',
                        'npz_dir': data_params['npz_dir'],
                        'years': data_params.get('years', None),
                        'n_channels': data_params.get('n_channels', 64),
                        'merge_patches': data_params.get('merge_patches', False),
                        'train_ratio': data_params.get('train_ratio', 0.7),
                        'val_ratio': data_params.get('val_ratio', 0.2)
                    })
                elif dataset_type == 'multiyear_h5':
                    creator_kwargs.update({
                        'dataset_type': 'multiyear_h5',
                        'h5_dir': data_params['h5_dir'],
                        'years': data_params.get('years', None),
                        'n_channels': data_params.get('n_channels', 64),
                        'merge_patches': data_params.get('merge_patches', False),
                        'train_ratio': data_params.get('train_ratio', 0.7),
                        'val_ratio': data_params.get('val_ratio', 0.2)
                    })
                elif dataset_type == 'single_modal':
                    creator_kwargs.update({
                        'dataset_type': 'single_modal',
                        'images_path': data_params['images_path'],
                        'masks_path': data_params['masks_path'],
                        'n_channels': data_params['n_channels'],
                        'year': data_params.get('year', 2019)
                    })
                else:
                    creator_kwargs.update({
                        'dataset_type': 'multi_modal',
                        'data_path': data_params['data_path'],
                        'mode': data_params.get('mode', 'random'),
                        'subset_ratio': data_params.get('subset_ratio', 1.0),
                        'csv_path': data_params.get('csv_path'),
                        'train_ratio': data_params.get('train_ratio', 0.75)
                    })

                subset_creator = SubsetCreator(**creator_kwargs)

                use_multi_modal_input = (dataset_type == 'multi_modal')
                mask_flag = data_params.get('add_glacier_mask', False)

                subset_data = subset_creator.create_subsets(
                    multi_modal_input=use_multi_modal_input,
                    resize=model_params.get('resize', 512),
                    batch_size=train_params.get('batch_size', 4),
                    use_augmentation=data_params.get('use_augmentation', True),
                    add_glacier_mask=mask_flag,
                    return_dist_map=data_params.get('return_dist_map', False),
                    num_workers=train_params.get('num_workers', 4)
                )

                train_loader = subset_data['loaders']['train']
                val_loader = subset_data['loaders']['val']
                self.logger.info("  Datasets ready.")

                model = self.create_model_from_config(exp_config_raw)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                pretrained_path = train_params.get('pretrained_path', None)
                finetune_mode = train_params.get('finetune_mode', None)

                if pretrained_path:
                    model = self.transfer_manager.load_pretrained(
                        model, pretrained_path, device, strict=False
                    )
                    model = self.transfer_manager.apply_finetune_mode(
                        model, finetune_mode, train_params
                    )

                if finetune_mode in ['layerwise_lr', 'layerwise_lr_gradual']:
                    optimizer = self.transfer_manager.create_optimizer_with_layerwise_lr(
                        model, train_params
                    )
                else:
                    optimizer = None

                unfreeze_scheduler = None
                if finetune_mode == 'gradual_unfreeze':
                    schedule = train_params.get('unfreeze_schedule', [])
                    if schedule:
                        unfreeze_scheduler = GradualUnfreezeScheduler(
                            model, schedule, self.logger
                        )

                exp_save_dir = self.output_dir / exp_id
                exp_save_dir.mkdir(parents=True, exist_ok=True)

                self.logger.info("  Starting training...")

                extra_train_kwargs = {}
                if optimizer is not None:
                    extra_train_kwargs['optimizer'] = optimizer
                if unfreeze_scheduler is not None:
                    extra_train_kwargs['unfreeze_scheduler'] = unfreeze_scheduler

                metrics = train_and_evaluate(
                    model,
                    train_loader,
                    val_loader,
                    exp_full_config,
                    exp_save_dir,
                    **extra_train_kwargs
                )

                result_entry = {
                    "experiment_id": exp_id,
                    "type": exp_config_raw.get('type', 'unknown'),
                    "priority": exp_config_raw.get('priority', 99),
                    "description": exp_config_raw.get('description', ''),
                    "finetune_mode": finetune_mode or 'scratch',
                    "pretrained": bool(pretrained_path),
                    **metrics
                }
                self.results.append(result_entry)
                self.logger.info(f"✅ Experiment {exp_id} finished. Final Val mIoU: {metrics.get('mIoU', 0.0):.4f}")

            except Exception as e:
                self.logger.error(f"❌ Experiment {exp_id} FAILED. Error: {e}", exc_info=True)

        self.save_results_and_report()

    def save_results_and_report(self):
        if not self.results:
            self.logger.warning("No results to save.")
            return

        df = pd.DataFrame(self.results)

        scratch_row = df[df['finetune_mode'] == 'scratch']
        if not scratch_row.empty:
            baseline_miou = scratch_row.iloc[0].get('mIoU', 0.0)
            df['mIoU_delta'] = df['mIoU'] - baseline_miou
            df['mIoU_improvement_%'] = (df['mIoU_delta'] / baseline_miou * 100).round(2)

        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")

        report_path = self.output_dir / "ablation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Transfer Learning Experiment Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for exp_type in sorted(df['type'].unique()):
                f.write(f"## {exp_type.upper()}\n\n")
                type_df = df[df['type'] == exp_type].sort_values(by='mIoU', ascending=False)

                display_cols = ['experiment_id', 'finetune_mode', 'mIoU', 'mIoU_delta', 'mIoU_improvement_%']
                display_cols = [c for c in display_cols if c in type_df.columns]

                f.write(type_df[display_cols].to_markdown(index=False))
                f.write("\n\n")

        self.logger.info(f"Report generated: {report_path}")