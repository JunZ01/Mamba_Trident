
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Any, List
from src.models.fusion_modules import SkipFusionNoDEM

class IdentityFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x1, x2):
        return x1, x2


class ModelModifier:
    @staticmethod
    def _set_module_by_path(model: nn.Module, path: str, new_module: nn.Module):
        path_parts = path.split('.')
        parent_module = model
        for part in path_parts[:-1]:
            parent_module = getattr(parent_module, part)
        child_name = path_parts[-1]
        setattr(parent_module, child_name, new_module)

    @staticmethod
    def apply_config_override(base_config: Dict, override: Dict) -> Dict:
        config = deepcopy(base_config)

        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        return _deep_update(config, override)

    @staticmethod
    def apply_module_replacement(model: nn.Module, replacements: List[Dict], model_config: Dict[str, Any]) -> nn.Module:
        model_copy = deepcopy(model)
        for rep_config in replacements:
            target_name = rep_config['target_module']
            replacement_name = rep_config['replacement_module']
            params_source = rep_config.get('params_source', 'none')

            for path, original_module in model_copy.named_modules():
                if original_module.__class__.__name__ == target_name:
                    new_module = None
                    if replacement_name == "torch.nn.Linear":
                        if params_source == "original":
                            # [Key fix] Smarter dimension inference logic
                            try:
                                path_parts = path.split('.')
                                decoder_index = int(path_parts[1])

                                dims = model_config['dims']
                                num_stages = len(dims)

                                out_dim = dims[num_stages - 2 - decoder_index]

                                use_synergy = model_config.get('use_synergy_skip', True)
                                num_inputs = 3 if use_synergy else 2
                                in_dim = num_inputs * out_dim

                                new_module = nn.Linear(in_dim, out_dim)
                            except Exception as e:
                                raise ValueError(
                                    f"Failed to infer dimensions for Linear replacement of {path}. Error: {e}")
                        else:
                            raise ValueError("Must specify params for Linear replacement if not from 'original'")

                    elif replacement_name == "ablation_engine.modifier.IdentityFusion":
                        new_module = IdentityFusion()

                    elif replacement_name == "src.models.fusion_modules.SkipFusionNoDEM":
                        if params_source == "original":
                            try:
                                path_parts = path.split('.')
                                stage_index = int(path_parts[1])  # path is "skip_fusers.0", "skip_fusers.1" ...
                                dim = model_config['dims'][stage_index]
                                num_inputs = sum([
                                    'dem' in model_config.get('in_channels', {}),
                                    's1' in model_config.get('in_channels', {}),
                                    's2' in model_config.get('in_channels', {})
                                ])
                                new_module = SkipFusionNoDEM(dim=dim, num_inputs=num_inputs)
                            except Exception as e:
                                raise ValueError(f"Failed to infer dims for SkipFusionNoDEM at {path}. Error: {e}")
                        else:
                            raise ValueError("SkipFusionNoDEM replacement requires params_source: 'original'")
                    else:
                        raise NotImplementedError(f"Replacement for '{replacement_name}' is not implemented.")

                    if new_module:
                        ModelModifier._set_module_by_path(model_copy, path, new_module)
                        print(f"  [Modifier] Replaced '{path}' ({target_name}) with {replacement_name}")

        return model_copy

    @staticmethod
    def apply_module_modifications(model: nn.Module, modifications: List[Dict]) -> nn.Module:
        model_copy = deepcopy(model)
        for mod_config in modifications:
            target_name = mod_config['target_module']
            action = mod_config['action']
            submodule_name = mod_config['submodule_name']
            if action == "remove_submodule":
                for path, module in model_copy.named_modules():
                    if module.__class__.__name__ == target_name:
                        if hasattr(module, submodule_name):
                            setattr(module, submodule_name, nn.Identity())
                            print(f"  [Modifier] Removed submodule '{submodule_name}' from '{path}' ({target_name})")
            else:
                raise NotImplementedError(f"Modification action '{action}' is not implemented.")
        return model_copy

    @staticmethod
    def apply_forward_hook(model: nn.Module, hook_config: Dict) -> nn.Module:
        model_copy = deepcopy(model)
        original_forward = model_copy.forward
        hook_type = hook_config['type']
        hook_params = hook_config['params']
        if hook_type == "modality_mask":
            def wrapped_forward(x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
                modified_input = deepcopy(x_dict)
                if not hook_params.get('dem', True): modified_input['dem'] = torch.zeros_like(x_dict['dem'])
                if not hook_params.get('s1', True): modified_input['s1'] = torch.zeros_like(
                    x_dict.get('s1') or x_dict.get('sar'))
                if not hook_params.get('s2', True): modified_input['s2'] = torch.zeros_like(
                    x_dict.get('s2') or x_dict.get('optical'))
                return original_forward(modified_input)

            model_copy.forward = wrapped_forward
            print(f"  [Modifier] Applied 'modality_mask' forward hook with params: {hook_params}")
        else:
            raise NotImplementedError(f"Forward hook type '{hook_type}' is not implemented.")
        return model_copy