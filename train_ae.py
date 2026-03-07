"""
Training script for the SDF Autoencoder (PVQVAE_diff / Diffusion-SDF AE).

This script trains the patch-wise variational autoencoder that encodes
64³ SDF volumes into a compact 8³ latent space used by the diffusion model.

Usage:
    python train_ae.py --data_root data/ShapeNet --cat all

    # Resume from epoch N:
    python train_ae.py --data_root data/ShapeNet --resume ckpt/vae_epoch-120.pth
                       --start_epoch 121

    # Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_ae.py --data_root data/ShapeNet --dist_train

Expected data layout:
    data/ShapeNet/sdf/<category_id>/<model_id>/pc_sdf_sample.h5
"""

import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.networks.pvqvae_networks.auto_encoder import PVQVAE_diff
from models.voxdiff.modules.losses.contperceptual import LPIPSWithDiscriminator
from models.voxdiff.util import instantiate_from_config
from models.voxdiff.data.snet import ShapeNetSDFDataset
from utils.util_3d import init_mesh_renderer, render_sdf


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Train SDF Autoencoder')

    # Data
    parser.add_argument('--data_root', type=str, default='data/ShapeNet',
                        help='Root directory for ShapeNet SDF data')
    parser.add_argument('--cat', type=str, default='all',
                        help='ShapeNet category (e.g. "chair") or "all"')
    parser.add_argument('--thres', type=float, default=0.2,
                        help='SDF clamp threshold')

    # Model config
    parser.add_argument('--vq_cfg', type=str, default='configs/voxdiff-uinu.yaml',
                        help='Path to config yaml containing first_stage_config')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=4.5e-6,
                        help='Base learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers per GPU')
    parser.add_argument('--val_split', type=float, default=0.05,
                        help='Fraction of data to use for validation')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch number (when resuming)')

    # Loss
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='Weight for KL divergence loss')
    parser.add_argument('--disc_start', type=int, default=50001,
                        help='Global step to start discriminator training')

    # Logging
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Print log every N steps')
    parser.add_argument('--vis_freq', type=int, default=500,
                        help='Visualise reconstructions every N steps')

    # Distributed
    parser.add_argument('--dist_train', action='store_true',
                        help='Enable distributed data-parallel training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank (set automatically by torchrun)')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_main_process(args):
    return (not args.dist_train) or (args.local_rank == 0)


def print_main(args, msg, color='white'):
    if is_main_process(args):
        cprint(msg, color)


def save_checkpoint(args, model, optimizer, epoch, global_step, label=None):
    if not is_main_process(args):
        return
    os.makedirs(args.save_dir, exist_ok=True)
    label = label or f'epoch-{epoch}'
    path = os.path.join(args.save_dir, f'vae_{label}.pth')
    state_dict = model.module.state_dict() if args.dist_train else model.state_dict()
    torch.save({
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }, path)
    cprint(f'[*] Saved checkpoint: {path}', 'green')


def load_checkpoint(args, model, optimizer=None):
    if not args.resume:
        return model, optimizer, args.start_epoch, 0
    if not os.path.exists(args.resume):
        cprint(f'[!] Checkpoint not found: {args.resume}', 'red')
        return model, optimizer, args.start_epoch, 0

    ckpt = torch.load(args.resume, map_location='cpu')

    # Support both bare state_dict and wrapped {'model': ...}
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
    cprint(f'[*] Loaded model weights from {args.resume}', 'blue')

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    epoch = ckpt.get('epoch', args.start_epoch)
    step = ckpt.get('global_step', 0)
    return model, optimizer, epoch, step


# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers  (mirror pvqvae_model_diff.py)
# ─────────────────────────────────────────────────────────────────────────────

from einops import rearrange


def unfold_to_cubes(x, cube_size=8, stride=8):
    """x: (B,1,64,64,64) → (B*P, 1, 8, 8, 8)"""
    x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
    x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
    x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')
    return x_cubes


def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
    """(B*P, c, d, h, w) → (B, c, D, H, W)"""
    x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
    x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                  p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Distributed setup ───────────────────────────────────────────────────
    if args.dist_train:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_main(args, f'[*] Using device: {device}', 'cyan')

    # ── Load model config ───────────────────────────────────────────────────
    config = OmegaConf.load(args.vq_cfg)
    fs_cfg = config.model.params.first_stage_config.params
    ddconfig = fs_cfg.ddconfig
    n_embed = fs_cfg.n_embed
    embed_dim = fs_cfg.embed_dim

    # Patch architecture parameters
    n_down = len(ddconfig.ch_mult) - 1
    cube_size = 2 ** n_down          # 8 for 4-level encoder
    ncubes_per_dim = ddconfig.resolution // cube_size  # 8 = 64/8

    # ── Build model ─────────────────────────────────────────────────────────
    model = PVQVAE_diff(ddconfig, n_embed, embed_dim)
    model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = LPIPSWithDiscriminator(
        disc_start=args.disc_start,
        kl_weight=args.kl_weight,
        disc_weight=0.0,        # no GAN loss for 3-D (2D discriminator not applicable)
        perceptual_weight=0.0,  # no 2-D perceptual loss for 3-D SDF
    ).to(device)

    # ── Optimiser ───────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9),
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    model, optimizer, start_epoch, global_step = load_checkpoint(args, model, optimizer)

    # ── Distributed wrapping ─────────────────────────────────────────────────
    if args.dist_train:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_dataset = ShapeNetSDFDataset(
        data_root=args.data_root,
        split='train',
        cat=args.cat,
        thres=args.thres,
    )
    if len(full_dataset) == 0:
        cprint('[!] Dataset is empty. Check --data_root path and data structure.', 'red')
        return

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print_main(args, f'[*] Dataset: {n_train} train / {n_val} val', 'cyan')

    if args.dist_train:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    for _ in range(start_epoch):
        scheduler.step()

    # ── Renderer (main process only) ──────────────────────────────────────
    renderer = None
    if is_main_process(args):
        try:
            renderer = init_mesh_renderer(image_size=256, dist=1.7, elev=20, azim=20, device=device)
        except Exception:
            renderer = None

    # ── Training loop ─────────────────────────────────────────────────────
    print_main(args, '[*] Starting AE training', 'green')
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.dist_train:
            train_sampler.set_epoch(epoch)

        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            x = batch['sdf'].to(device)               # (B, 1, 64, 64, 64)
            cur_bs = x.shape[0]

            # Unfold into patches
            x_cubes = unfold_to_cubes(x, cube_size, cube_size)  # (B*P, 1, 8, 8, 8)

            # Forward
            posterior = model.encode(x_cubes) if not args.dist_train \
                else model.module.encode(x_cubes)
            z = posterior.sample()
            z_voxel = fold_to_voxels(z, batch_size=cur_bs, ncubes_per_dim=ncubes_per_dim)
            dec = (model.decode if not args.dist_train else model.module.decode)(z_voxel)

            # Loss (optimizer_idx=0 → AE loss)
            last_layer = (model.get_last_layer if not args.dist_train
                          else model.module.get_last_layer)()
            aeloss, log_dict = loss_fn(
                inputs=x,
                reconstructions=dec,
                posteriors=posterior,
                optimizer_idx=0,
                global_step=global_step,
                last_layer=last_layer,
                split='train',
            )

            # Backward
            optimizer.zero_grad(set_to_none=True)
            aeloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            epoch_losses.append(aeloss.item())

            if is_main_process(args) and step % args.log_freq == 0:
                rec_loss = log_dict.get('train/rec_loss', torch.tensor(0.0))
                kl_loss = log_dict.get('train/kl_loss', torch.tensor(0.0))
                nll_loss = log_dict.get('train/nll_loss', torch.tensor(0.0))
                elapsed = time.time() - t0
                print(f'Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(train_loader)}] '
                      f'Loss: {aeloss.item():.4f} '
                      f'(rec={_to_scalar(rec_loss):.4f}, '
                      f'kl={_to_scalar(kl_loss):.6f}, '
                      f'nll={_to_scalar(nll_loss):.4f}) '
                      f'| {elapsed:.1f}s elapsed')

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        val_loss = validate(args, model, val_loader, loss_fn, device, cube_size, ncubes_per_dim, global_step)
        avg_train = sum(epoch_losses) / max(len(epoch_losses), 1)
        print_main(args, f'[Epoch {epoch+1}] train_loss={avg_train:.4f}  val_loss={val_loss:.4f}', 'yellow')

        # ── Checkpoint ─────────────────────────────────────────────────────
        if is_main_process(args):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args, model, optimizer, epoch + 1, global_step, label='best')

            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint(args, model, optimizer, epoch + 1, global_step,
                                label=f'epoch-{epoch+1}')

    # ── Final checkpoint ─────────────────────────────────────────────────
    save_checkpoint(args, model, optimizer, args.epochs, global_step, label='final')
    print_main(args, '[*] Training complete.', 'green')

    if args.dist_train:
        dist.destroy_process_group()


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


@torch.no_grad()
def validate(args, model, val_loader, loss_fn, device, cube_size, ncubes_per_dim, global_step):
    model.eval()
    losses = []
    for batch in val_loader:
        x = batch['sdf'].to(device)
        cur_bs = x.shape[0]
        x_cubes = unfold_to_cubes(x, cube_size, cube_size)

        posterior = (model.encode if not args.dist_train else model.module.encode)(x_cubes)
        z = posterior.mode()  # deterministic at validation
        z_voxel = fold_to_voxels(z, batch_size=cur_bs, ncubes_per_dim=ncubes_per_dim)
        dec = (model.decode if not args.dist_train else model.module.decode)(z_voxel)

        last_layer = (model.get_last_layer if not args.dist_train
                      else model.module.get_last_layer)()
        loss, _ = loss_fn(
            inputs=x,
            reconstructions=dec,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=global_step,
            last_layer=last_layer,
            split='val',
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / max(len(losses), 1)


if __name__ == '__main__':
    main()
