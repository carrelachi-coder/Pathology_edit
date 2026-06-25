import argparse
import os
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Original Cross V1 checkpoint dir')
    parser.add_argument('--output-dir', required=True, help='Output dir for clean checkpoint')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading original checkpoint from: ' + str(checkpoint_dir))
    print('Output clean checkpoint to: ' + str(output_dir))

    # 1. Copy config.json and diffusion model (no changes)
    print('Copying config.json...')
    os.system('cp ' + str(checkpoint_dir / 'config.json') + ' ' + str(output_dir / 'config.json'))
    
    print('Copying diffusion_pytorch_model.safetensors...')
    os.system('cp ' + str(checkpoint_dir / 'diffusion_pytorch_model.safetensors') + ' ' + str(output_dir / 'diffusion_pytorch_model.safetensors'))

    # 2. Process conditioning weights (remove ref_encoder)
    print('Processing conditioning weights...')
    cond_state = torch.load(checkpoint_dir / 'phase5_conditioning.pt', map_location='cpu', weights_only=False)
    clean_cond_state = {
        'hte': cond_state['hte'],
        'tissue_downsampler': cond_state['tissue_downsampler'],
        'nuclei_encoder': cond_state['nuclei_encoder'],
    }
    torch.save(clean_cond_state, output_dir / 'phase5_conditioning.pt')
    print('  - Kept: hte, tissue_downsampler, nuclei_encoder')
    print('  - Removed: ref_encoder_* weights')

    # 3. Create README
    readme = '# Cross V1 Checkpoint (NO IP-Adapter Clean Version)\n\nThis is a clean version of the Cross V1 checkpoint with:\n- IP-Adapter attention layers REMOVED\n- UNI reference image encoder REMOVED\n\nUsage: Use with pipeline_cross_v1_no_ip.py for inference.\n'
    with open(output_dir / 'README_NO_IP.txt', 'w') as f:
        f.write(readme)

    print('\n✅ Done! Clean checkpoint exported.')

if __name__ == '__main__':
    main()
