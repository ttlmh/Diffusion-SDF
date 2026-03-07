"""
Shape Manipulation inference script for Diffusion-SDF.

Given an existing 3D shape (SDF) and a text prompt, this script modifies
the shape according to the text description using the SDEdit approach:
  1. Encode the input shape to the latent space.
  2. Add Gaussian noise up to a chosen timestep (controls how much to change).
  3. Denoise with the new text prompt to guide the manipulation.

Usage:
    # Moderate manipulation (t=500 / 1000):
    python shape_manipulation.py \\
        --input_sdf path/to/shape.h5 \\
        --prompt "a chair with a cushion" \\
        --strength 0.5

    # Strong manipulation (t=750 / 1000 – more creative freedom):
    python shape_manipulation.py \\
        --input_sdf path/to/shape.h5 \\
        --prompt "a modern minimalist chair" \\
        --strength 0.75

Arguments:
    --input_sdf:  Path to h5 file with the original SDF (float32, 64³).
    --prompt:     Text description for the target modification.
    --strength:   Noise strength in [0, 1]. Higher = more modification.
                  0.0 → identity (no change), 1.0 → full generation.
    --n_samples:  Number of manipulated variants to generate.
"""

import argparse
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.io import IO
from tqdm import tqdm

from models.voxdiff.models.diffusion.ddim import DDIMSampler
from models.voxdiff.util import instantiate_from_config
from utils.qual_util import save_mesh_as_gif
from utils.util_3d import init_mesh_renderer, sdf_to_mesh


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sdf(path: str, thres: float = 0.2) -> torch.Tensor:
    """Load and normalise an SDF from an h5 file. Returns (1,1,64,64,64)."""
    with h5py.File(path, 'r') as f:
        if 'pc_sdf_sample' in f:
            sdf = f['pc_sdf_sample'][:].astype(np.float32)
        elif 'sdf' in f:
            sdf = f['sdf'][:].astype(np.float32)
        else:
            key = list(f.keys())[0]
            sdf = f[key][:].astype(np.float32)
    sdf = torch.from_numpy(sdf).view(1, 1, 64, 64, 64)
    sdf = torch.clamp(sdf, -thres, thres) / thres
    return sdf


@torch.no_grad()
def sdedit(
    model,
    sampler: DDIMSampler,
    x0_latent: torch.Tensor,
    conditioning,
    unconditional_conditioning,
    strength: float,
    ddim_steps: int = 50,
    eta: float = 0.0,
    scale: float = 5.0,
) -> torch.Tensor:
    """
    SDEdit: noise x0_latent to timestep t = strength * T, then denoise with text.

    Args:
        model:                   VoxelizedDiffusion model.
        sampler:                 DDIMSampler instance.
        x0_latent:               Encoded latent of the original shape (B, C, H, W, D).
        conditioning:            Text conditioning tensor.
        unconditional_conditioning: Null conditioning for CFG.
        strength:                Noise level in [0, 1].
        ddim_steps:              Total DDIM steps.
        eta:                     DDIM stochasticity (0 = deterministic).
        scale:                   CFG scale.

    Returns:
        Denoised latent tensor of same shape as x0_latent.
    """
    device = x0_latent.device

    # Total diffusion steps
    T = model.num_timesteps  # typically 1000

    # The DDIM schedule
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_discretize='slow_fast',
                          ddim_eta=eta, verbose=False)

    # Determine the starting timestep for denoising
    # strength=0.5 → start from step ddim_steps//2 in the DDIM schedule
    t_start = min(int(strength * ddim_steps), ddim_steps - 1)
    t_start = max(t_start, 1)

    # Corresponding diffusion timestep
    t_enc_idx = len(sampler.ddim_timesteps) - t_start
    t_enc = sampler.ddim_timesteps[t_enc_idx]
    t_tensor = torch.full((x0_latent.shape[0],), t_enc, device=device, dtype=torch.long)

    # Add noise to the latent at this timestep
    noise = torch.randn_like(x0_latent)
    x_noisy = model.q_sample(x_start=x0_latent, t=t_tensor, noise=noise)

    # Run DDIM sampling, starting from the noised latent (not pure Gaussian)
    shape = list(x0_latent.shape[1:])  # [C, H, W, D]

    # We pass x_noisy as x_T and use only t_start DDIM steps
    samples, _ = sampler.sample(
        S=t_start,
        conditioning=conditioning,
        batch_size=x0_latent.shape[0],
        shape=shape,
        verbose=False,
        x_T=x_noisy,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=unconditional_conditioning,
        eta=eta,
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion-SDF: Shape Manipulation')
    parser.add_argument('--input_sdf', type=str, default='',
                        help='Path to h5 file with the original SDF. '
                             'If empty, a random shape is used as demo.')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text description for the manipulated shape.')
    parser.add_argument('--out_path', type=str, default='outputs/shape_manipulation',
                        help='Output directory.')
    parser.add_argument('--config_path', type=str,
                        default='configs/voxdiff-uinu.yaml')
    parser.add_argument('--model_path', type=str,
                        default='ckpt/voxdiff-uinu.ckpt')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of manipulated variants to generate.')
    parser.add_argument('--strength', type=float, default=0.5,
                        help='Manipulation strength in [0,1]. '
                             '0=no change, 1=full generation.')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=5.0,
                        help='Classifier-free guidance scale.')
    parser.add_argument('--thres', type=float, default=0.2,
                        help='SDF normalisation threshold.')
    parser.add_argument('--save_obj', action='store_true',
                        help='Also save OBJ mesh files.')
    return parser.parse_args()


def main():
    opt = parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    config = OmegaConf.load(opt.config_path)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    dist, elev, azim = 1.7, 20, 20
    renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=device)

    sampler = DDIMSampler(model)
    os.makedirs(opt.out_path, exist_ok=True)

    # ── Load / create input SDF ───────────────────────────────────────────
    if opt.input_sdf and os.path.exists(opt.input_sdf):
        sdf = load_sdf(opt.input_sdf, thres=opt.thres)  # (1, 1, 64, 64, 64)
        print(f'[*] Loaded input SDF from {opt.input_sdf}')
    else:
        print('[*] No input SDF provided; using a random SDF as demo.')
        sdf = torch.randn(1, 1, 64, 64, 64) * 0.1

    sdf = sdf.to(device)

    # Replicate for batch
    sdf_batch = sdf.expand(opt.n_samples, -1, -1, -1, -1)  # (n_samples, 1, 64, 64, 64)

    with torch.no_grad():
        with model.ema_scope():
            # ── Encode input to latent ─────────────────────────────────────
            posterior = model.first_stage_model.encode_whole(sdf_batch)
            x0_latent = model.get_first_stage_encoding(posterior)
            # x0_latent: (n_samples, 8, 8, 8, 8)

            # ── Text conditioning ─────────────────────────────────────────
            prompts = opt.n_samples * [opt.prompt]
            c = model.get_learned_conditioning(prompts)

            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [''])

            # ── SDEdit manipulation ───────────────────────────────────────
            samples_ddim = sdedit(
                model=model,
                sampler=sampler,
                x0_latent=x0_latent,
                conditioning=c,
                unconditional_conditioning=uc,
                strength=opt.strength,
                ddim_steps=opt.ddim_steps,
                eta=opt.ddim_eta,
                scale=opt.scale,
            )
            # samples_ddim: (n_samples, 8, 8, 8, 8)

            # ── Decode to SDF ─────────────────────────────────────────────
            manipulated_sdf = model.decode_first_stage(samples_ddim)
            # (n_samples, 1, 64, 64, 64)

    # ── Save results ──────────────────────────────────────────────────────
    tag = opt.prompt.replace(' ', '-')
    strength_tag = f's{int(opt.strength * 100)}'
    out_dir = os.path.join(opt.out_path, f'{tag}_{strength_tag}')
    os.makedirs(out_dir, exist_ok=True)

    # Save the original shape for comparison
    orig_mesh = sdf_to_mesh(sdf)
    if orig_mesh is not None:
        orig_gif = os.path.join(out_dir, 'original.gif')
        save_mesh_as_gif(renderer, orig_mesh, nrow=1, out_name=orig_gif)
        print(f'[*] Saved original GIF: {orig_gif}')

    # Save the manipulated shapes
    gen_mesh = sdf_to_mesh(manipulated_sdf)
    if gen_mesh is not None:
        out_gif = os.path.join(out_dir, f'manipulated_{tag}.gif')
        save_mesh_as_gif(renderer, gen_mesh, nrow=2, out_name=out_gif)
        print(f'[*] Saved manipulated GIF: {out_gif}')

        if opt.save_obj:
            for k, mesh in enumerate(gen_mesh):
                obj_path = os.path.join(out_dir, f'manipulated_{tag}_{k}.obj')
                IO().save_mesh(mesh, obj_path)
                print(f'[*] Saved OBJ: {obj_path}')

    print('[*] Shape manipulation done.')


if __name__ == '__main__':
    main()
