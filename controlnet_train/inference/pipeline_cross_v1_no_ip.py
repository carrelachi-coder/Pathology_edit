"""
Clean Cross V1 inference pipeline WITHOUT IP-Adapter and UNI reference encoder.
For research release - keeps only the essential components.

Usage:
    from controlnet_train.inference.pipeline_cross_v1_no_ip import load_cross_v1_no_ip_bundle
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, NamedTuple
from diffusers import FluxControlNetModel, FluxControlNetPipeline


class CrossV1NoIPInferenceBundle(NamedTuple):
    pretrained_model_name_or_path: str
    checkpoint_path: Path
    device: str
    torch_dtype: torch.dtype
    flux_pipeline: FluxControlNetPipeline
    controlnet: Any
    condition_modules: dict[str, nn.Module]
    control_spec: Any


def patch_controlnet_x_embedder(controlnet: nn.Module, packed_control_channels: int) -> nn.Module:
    old_x_embedder = controlnet.controlnet_x_embedder
    if old_x_embedder.in_features == packed_control_channels:
        return controlnet
    new_x_embedder = nn.Linear(packed_control_channels, old_x_embedder.out_features)
    with torch.no_grad():
        new_x_embedder.weight.zero_()
        copy_width = min(old_x_embedder.in_features, packed_control_channels)
        new_x_embedder.weight[:, :copy_width] = old_x_embedder.weight[:, :copy_width]
        if old_x_embedder.bias is not None:
            new_x_embedder.bias.copy_(old_x_embedder.bias)
    controlnet.controlnet_x_embedder = new_x_embedder
    return controlnet


def load_cross_v1_no_ip_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = 'cuda',
    torch_dtype: torch.dtype | None = None,
) -> CrossV1NoIPInferenceBundle:
    checkpoint = Path(checkpoint_path)
    
    # Detect dtype
    if torch_dtype is None:
        if device == 'cuda' and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print(f'[CrossV1-NoIP] Loading clean model on {device} with {torch_dtype}...')
    print(f'  - NO IP-Adapter attention layers')
    print(f'  - NO UNI reference image encoder')
    
    # ---- 1. Load ControlNet ----
    controlnet_config = FluxControlNetModel.load_config(checkpoint)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, packed_control_channels=640)  # Cross V1 uses 640
    
    # Load weights
    from safetensors.torch import load_file
    state_dict = load_file(str(checkpoint / 'diffusion_pytorch_model.safetensors'))
    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  - Warning: {len(missing)} missing keys')
    if unexpected:
        print(f'  - Warning: {len(unexpected)} unexpected keys')
    controlnet.to(dtype=torch_dtype)
    
    # ---- 2. Load FLUX pipeline ----
    pipe = FluxControlNetPipeline.from_pretrained(
        str(pretrained_model_name_or_path),
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # ---- 3. Load ONLY condition modules (NO reference encoder) ----
    import sys
    sys.path.insert(0, '.')
    from controlnet_train.modules import HierarchicalTissueEmbedding
    from controlnet_train.modules import TissueConditionDownsampler, NucleiConditionEncoder
    
    cond_state = torch.load(checkpoint / 'phase5_conditioning.pt', map_location='cpu', weights_only=False)
    hte_state = cond_state['hte']
    tissue_state = cond_state['tissue_downsampler']
    nuclei_state = cond_state['nuclei_encoder']
    
    def _count_conv_blocks(state_dict, prefix):
        return len({int(key.split('.')[1]) for key in state_dict if key.startswith(prefix) and key.endswith('block.0.weight')})
    
    hte_dim = hte_state['parent_embeddings.weight'].shape[1]
    tissue_in = tissue_state['blocks.0.block.0.weight'].shape[1]
    tissue_hidden = tissue_state['blocks.0.block.0.weight'].shape[0]
    tissue_out = tissue_state[f'blocks.{_count_conv_blocks(tissue_state, "blocks") - 1}.block.0.weight'].shape[0]
    nuclei_embed = nuclei_state['embedding.weight'].shape[1]
    nuclei_out = nuclei_state['downsampler.0.block.0.weight'].shape[0]
    nuclei_blocks = _count_conv_blocks(nuclei_state, 'downsampler')
    
    modules = {
        'hte': HierarchicalTissueEmbedding(embedding_dim=hte_dim),
        'tissue_downsampler': TissueConditionDownsampler(
            in_channels=tissue_in,
            hidden_channels=tissue_hidden,
            out_channels=tissue_out,
            num_blocks=_count_conv_blocks(tissue_state, 'blocks'),
        ),
        'nuclei_encoder': NucleiConditionEncoder(
            embedding_dim=nuclei_embed,
            out_channels=nuclei_out,
            num_blocks=nuclei_blocks,
        ),
    }
    
    for name, module in modules.items():
        module.load_state_dict(cond_state[name])
        module.to(device=device, dtype=torch_dtype)
        module.eval()
    
    # Simple control spec
    class SimpleControlSpec:
        packed_channels = 160  # raw channels
        packed_control_channels = 640  # final input dim
        num_classes = 16
        num_nuclei_classes = 8
    
    print('[CrossV1-NoIP] Loaded successfully!')
    print(f'  - Condition modules: {list(modules.keys())}')
    print(f'  - Memory saved: ~12GB (UNI) + ~300MB (IP-adapter) = ~12.3GB')
    
    return CrossV1NoIPInferenceBundle(
        pretrained_model_name_or_path=str(pretrained_model_name_or_path),
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=torch_dtype,
        flux_pipeline=pipe,
        controlnet=controlnet,
        condition_modules=modules,
        control_spec=SimpleControlSpec(),
    )


if __name__ == '__main__':
    bundle = load_cross_v1_no_ip_bundle(
        pretrained_model_name_or_path='/data/huggingface/FLUX.1-dev',
        checkpoint_path='/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1_no_ip',
        device='cuda',
    )
    print('✅ Test passed!')
