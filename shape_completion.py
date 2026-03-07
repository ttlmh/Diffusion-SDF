"""
Shape Completion inference script for Diffusion-SDF.

Given a partial/incomplete SDF and a text prompt, this script completes
the missing regions using the trained Voxelized Diffusion model.

Usage:
    python shape_completion.py \\
        --input_sdf path/to/partial.h5 \\
        --prompt "a wooden chair" \\
        --mask_axis z --mask_ratio 0.5

    # Mask with a custom threshold on SDF values:
    python shape_completion.py \\
        --input_sdf path/to/shape.h5 \\
        --prompt "a dining table" \\
        --mask_type threshold --mask_value 0.0

Arguments:
    --input_sdf:    Path to h5 file containing the partial SDF (float32, 64³).
                    If not provided, a random noise shape is used as a demo.
    --prompt:       Text description of the complete shape.
    --mask_axis:    Axis to cut ('x', 'y', 'z').
    --mask_ratio:   Fraction of the volume to mask out (0–1).
    --mask_type:    'half' (axial cut) or 'threshold' (SDF-value based).
    --mask_value:   SDF threshold for 'threshold' masking.
    --n_samples:    Number of completions to generate.
"""

import argparse
import os

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch3d.io import IO
from tqdm import trange

from models.voxdiff.models.diffusion.ddim import DDIMSampler
from models.voxdiff.util import instantiate_from_config
from utils.qual_util import save_mesh_as_gif
from utils.util_3d import init_mesh_renderer, sdf_to_mesh, read_sdf


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
    sdf = torch.from_numpy(sdf).reshape(1, 1, 64, 64, 64)
    sdf = torch.clamp(sdf, -thres, thres) / thres
    return sdf


def make_axial_mask(sdf_shape, axis: str, ratio: float, device) -> torch.Tensor:
    """
    Create a binary mask for an axial cut.

    Mask = 1 → keep this voxel (known region).
    Mask = 0 → this region is unknown / to be completed.

    Args:
        sdf_shape:  (B, C, D, H, W)
        axis:       'x', 'y', or 'z'
        ratio:      fraction of the axis to mask (0.5 = mask the second half)
        device:     torch device

    Returns:
        sdf_mask:  (B, 1, D, H, W) float, 1=known 0=unknown
    """
    B, C, D, H, W = sdf_shape
    axis_map = {'x': 2, 'y': 3, 'z': 4}
    dim = axis_map[axis]
    sizes = [D, H, W]
    split = int(sizes[dim - 2] * (1.0 - ratio))

    mask = torch.ones(B, 1, D, H, W, device=device)
    slices = [slice(None)] * 5
    slices[dim] = slice(split, None)
    mask[tuple(slices)] = 0.0
    return mask


def make_threshold_mask(sdf: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Mark voxels with SDF < threshold as 'known' (surface / interior) and
    voxels with SDF >= threshold as 'unknown' (to be completed).

    Returns mask (B, 1, D, H, W) float.
    """
    mask = (sdf < threshold).float()
    return mask


def sdf_mask_to_latent_mask(sdf_mask: torch.Tensor, latent_size: int = 8) -> torch.Tensor:
    """
    Downsample a 64³ SDF mask to the latent space size (default 8³) via max-pooling.

    Any latent voxel that overlaps with an unknown SDF voxel is set to unknown.
    sdf_mask: (B, 1, 64, 64, 64) ∈ {0, 1}
    Returns: (B, 1, latent_size, latent_size, latent_size) ∈ {0, 1}
    """
    scale = sdf_mask.shape[-1] // latent_size
    # Use avg pooling; threshold at 1.0 so any partial unknown voxel → unknown
    pooled = torch.nn.functional.avg_pool3d(
        sdf_mask.float(),
        kernel_size=scale,
        stride=scale,
    )
    # If avg < 1.0, at least one SDF voxel was unknown → latent unknown
    latent_mask = (pooled >= 1.0).float()
    return latent_mask


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Diffusion-SDF: Shape Completion')
    parser.add_argument('--input_sdf', type=str, default='',
                        help='Path to h5 file with partial SDF. '
                             'If empty, a demo random shape is used.')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text description of the target shape.')
    parser.add_argument('--out_path', type=str, default='outputs/shape_completion',
                        help='Output directory.')
    parser.add_argument('--config_path', type=str,
                        default='configs/voxdiff-uinu.yaml')
    parser.add_argument('--model_path', type=str,
                        default='ckpt/voxdiff-uinu.ckpt')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of completions to generate.')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=5.0,
                        help='Classifier-free guidance scale.')
    parser.add_argument('--thres', type=float, default=0.2,
                        help='SDF normalisation threshold.')

    # Masking options
    parser.add_argument('--mask_type', type=str, default='half',
                        choices=['half', 'threshold'],
                        help='"half": axial cut; "threshold": SDF-value based.')
    parser.add_argument('--mask_axis', type=str, default='z',
                        choices=['x', 'y', 'z'])
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Fraction of the axis to mask (for "half" mode).')
    parser.add_argument('--mask_value', type=float, default=0.0,
                        help='SDF threshold for "threshold" masking.')
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

    # ── Load / create partial SDF ─────────────────────────────────────────
    if opt.input_sdf and os.path.exists(opt.input_sdf):
        sdf = load_sdf(opt.input_sdf, thres=opt.thres)  # (1, 1, 64, 64, 64)
        print(f'[*] Loaded partial SDF from {opt.input_sdf}')
    else:
        print('[*] No input SDF provided; using a random SDF as demo.')
        sdf = torch.randn(1, 1, 64, 64, 64) * 0.1  # near-flat surface

    sdf = sdf.to(device)

    # ── Build mask ────────────────────────────────────────────────────────
    if opt.mask_type == 'half':
        sdf_mask = make_axial_mask(
            sdf.shape, axis=opt.mask_axis, ratio=opt.mask_ratio, device=device
        )  # (1, 1, 64, 64, 64)
    else:
        sdf_mask = make_threshold_mask(sdf, threshold=opt.mask_value).to(device)

    # The masked SDF (only known regions)
    partial_sdf = sdf * sdf_mask

    # ── Encode the known region into latent space ─────────────────────────
    with torch.no_grad():
        with model.ema_scope():
            # Encode the (full) partial SDF
            posterior = model.first_stage_model.encode_whole(
                partial_sdf.expand(opt.n_samples, -1, -1, -1, -1)
            )
            x0_latent = model.get_first_stage_encoding(posterior)
            # x0_latent: (n_samples, 8, 8, 8, 8)

            # Build matching latent mask
            latent_mask = sdf_mask_to_latent_mask(
                sdf_mask.expand(opt.n_samples, -1, -1, -1, -1),
                latent_size=x0_latent.shape[-1],
            )  # (n_samples, 1, 8, 8, 8)
            # Expand to match latent channels
            latent_mask = latent_mask.expand_as(x0_latent)  # (n_samples, 8, 8, 8, 8)

            # ── Text conditioning ─────────────────────────────────────────
            prompts = opt.n_samples * [opt.prompt]
            c = model.get_learned_conditioning(prompts)

            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [''])

            # ── DDIM sampling with mask ───────────────────────────────────
            shape = list(x0_latent.shape[1:])  # [8, 8, 8, 8]

            samples_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                mask=latent_mask,          # 1 = known (lock to x0), 0 = generate
                x0=x0_latent,             # the encoded known latent
            )

            # Decode completed latent → SDF
            completed_sdf = model.decode_first_stage(samples_ddim)
            # completed_sdf: (n_samples, 1, 64, 64, 64)

    # ── Save results ──────────────────────────────────────────────────────
    tag = opt.prompt.replace(' ', '-')
    out_dir = os.path.join(opt.out_path, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Also save the partial input for visual comparison
    gen_mesh = sdf_to_mesh(completed_sdf)
    gif_path = os.path.join(out_dir, f'completion_{tag}.gif')
    save_mesh_as_gif(renderer, gen_mesh, nrow=2, out_name=gif_path)
    print(f'[*] Saved GIF: {gif_path}')

    if opt.save_obj and gen_mesh is not None:
        for k, mesh in enumerate(gen_mesh):
            obj_path = os.path.join(out_dir, f'completion_{tag}_{k}.obj')
            IO().save_mesh(mesh, obj_path)
            print(f'[*] Saved OBJ: {obj_path}')

    # Also render the partial input
    partial_mesh = sdf_to_mesh(partial_sdf)
    if partial_mesh is not None:
        partial_gif = os.path.join(out_dir, 'partial_input.gif')
        save_mesh_as_gif(renderer, partial_mesh, nrow=1, out_name=partial_gif)
        print(f'[*] Saved partial input GIF: {partial_gif}')

    print('[*] Shape completion done.')


if __name__ == '__main__':
    main()
