"""
Main training script for the Voxelized Diffusion model (Diffusion-SDF).

Trains the text-conditioned 3D diffusion model using PyTorch Lightning.
The autoencoder (AE) must be pre-trained first using train_ae.py.

Usage:
    python main.py --config configs/voxdiff-uinu.yaml

    # Resume from checkpoint:
    python main.py --config configs/voxdiff-uinu.yaml --resume /path/to/checkpoint.ckpt

    # Multi-GPU:
    python main.py --config configs/voxdiff-uinu.yaml --gpus 0,1,2,3
"""

import argparse
import datetime
import os
import signal
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_obj_from_str(string, reload=False):
    """Instantiate a class or function from a dotted string."""
    module, cls = string.rsplit('.', 1)
    if reload:
        import importlib
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(__import__(module, fromlist=[cls]), cls)


def instantiate_from_config(config):
    if 'target' not in config:
        if config == '__is_first_stage__':
            return None
        if config == '__is_unconditional__':
            return None
        raise KeyError('Expected key "target" to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', {}))


# ─────────────────────────────────────────────────────────────────────────────
# Data Module
# ─────────────────────────────────────────────────────────────────────────────

class WrappedDataset(Dataset):
    """Wraps an arbitrary mapping to be compatible with DataLoader."""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    """
    LightningDataModule that instantiates train/val/test datasets from config.

    Expected config format:
        data:
          target: main.DataModuleFromConfig
          params:
            batch_size: 48
            num_workers: 5
            train:
              target: models.voxdiff.data.snet.ShapeNetTextDatasetTrain
              params: {thres: 0.2, ucond_p: 0.2, cat: all}
            validation:
              target: models.voxdiff.data.snet.ShapeNetTextDatasetValidation
              params: {thres: 0.2, cat: all}
    """

    def __init__(
        self,
        batch_size: int,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap: bool = False,
        num_workers: int = None,
        shuffle_test_loader: bool = False,
        use_worker_init_fn: bool = False,
        shuffle_val_dataloader: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
        if predict is not None:
            self.dataset_configs['predict'] = predict

        self.wrap = wrap
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_dataloader = shuffle_val_dataloader

    def prepare_data(self):
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = {}
        for k, cfg in self.dataset_configs.items():
            self.datasets[k] = instantiate_from_config(cfg)
            if self.wrap:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _make_loader(self, split, shuffle=False):
        dataset = self.datasets[split]
        is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable else shuffle,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_loader('train', shuffle=True)

    def val_dataloader(self):
        return self._make_loader('validation', shuffle=self.shuffle_val_dataloader)

    def test_dataloader(self):
        return self._make_loader('test', shuffle=self.shuffle_test_loader)

    def predict_dataloader(self):
        return self._make_loader('predict', shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class SetupCallback(Callback):
    """Create output directories and save configs before training starts."""

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print('Received KeyboardInterrupt signal; attempting to create checkpoint ...')
            ckpt_path = os.path.join(self.ckptdir, 'last.ckpt')
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, f'{self.now}-project.yaml'))
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}),
                           os.path.join(self.cfgdir, f'{self.now}-lightning.yaml'))


class ImageLogger(Callback):
    """Periodically log reconstructed and generated shape renderings."""

    def __init__(self, batch_frequency=2000, max_images=4, clamp=True,
                 increase_log_steps=True, rescale=True, disabled=False,
                 log_on_batch_idx=False, log_first_step=False, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        import torchvision
        root = os.path.join(save_dir, 'images', split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]
            grid = grid.permute(1, 2, 0)  # H W C
            grid_np = (grid.numpy() * 255).clip(0, 255).astype(np.uint8)
            filename = f'{k}_gs-{global_step:06d}_e-{current_epoch:06d}_b-{batch_idx:06d}.png'
            import PIL.Image
            PIL.Image.fromarray(grid_np).save(os.path.join(root, filename))

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(
                    batch,
                    split=split,
                    **self.log_images_kwargs,
                )
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            self.log_local(
                pl_module.logger.save_dir,
                split, images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or
                (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split='val')
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_grad_norm(trainer, pl_module, batch, batch_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-c', '--config',
        nargs='*',
        metavar='config.yaml',
        help='paths to base configs. Loaded from left-to-right. '
             'Parameters can be overwritten or added with command-line options of the form `--key value`.',
        default=['configs/voxdiff-uinu.yaml'],
    )
    parser.add_argument(
        '-r', '--resume',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='resume from logdir or checkpoint in logdir',
    )
    parser.add_argument(
        '-l', '--logdir',
        type=str,
        default='logs',
        help='directory for logging',
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f', '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )
    parser.add_argument(
        '-n', '--name',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='postfix for logdir',
    )
    parser.add_argument(
        '--no_test',
        action='store_true',
        default=False,
        help='disable test',
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='gpu ids to use, e.g. "0,1,2"',
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=500,
        help='maximum number of training epochs',
    )
    parser.add_argument(
        '--scale_lr',
        action='store_true',
        default=False,
        help='scale base-lr by ngpu * batch_size * n_accumulate',
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # ── parse arguments ──────────────────────────────────────────────────────
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # ── seed ─────────────────────────────────────────────────────────────────
    seed_everything(opt.seed)

    # ── load configs ─────────────────────────────────────────────────────────
    configs = [OmegaConf.load(c) for c in opt.config]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop('lightning', OmegaConf.create())
    trainer_config = lightning_config.get('trainer', OmegaConf.create())

    # ── resolve GPU configuration ─────────────────────────────────────────────
    gpu_ids = [int(g) for g in str(opt.gpus).split(',')]
    n_gpus = len(gpu_ids)
    if n_gpus > 1:
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = gpu_ids
        trainer_config['strategy'] = 'ddp'
    else:
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = gpu_ids

    # ── logging directory ─────────────────────────────────────────────────────
    if opt.name:
        name = opt.name
    elif opt.resume:
        name = opt.resume.split('/')[-1].split('.')[0]
    else:
        cfg_name = '-'.join(os.path.splitext(os.path.basename(c))[0]
                            for c in opt.config)
        name = cfg_name
    if opt.postfix:
        name = f'{name}_{opt.postfix}'

    nowname = f'{now}_{name}'
    logdir = os.path.join(opt.logdir, nowname)
    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')

    # ── resume ────────────────────────────────────────────────────────────────
    ckpt_path = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f'Cannot find {opt.resume}')
        if os.path.isfile(opt.resume):
            ckpt_path = opt.resume
            logdir = '/'.join(opt.resume.split('/')[:-2])
        else:
            # Directory: find latest checkpoint
            ckptdir = os.path.join(opt.resume, 'checkpoints')
            ckpt_path = os.path.join(ckptdir, 'last.ckpt')

    # ── scale learning rate ───────────────────────────────────────────────────
    base_lr = config.model.base_learning_rate
    if opt.scale_lr:
        bs = config.data.params.batch_size
        accumulate = trainer_config.get('accumulate_grad_batches', 1)
        lr = accumulate * n_gpus * bs * base_lr
        print(f'Setting learning rate to {lr:.2e} = '
              f'{accumulate} (accumulate) × {n_gpus} (GPUs) × '
              f'{bs} (bs) × {base_lr:.2e} (base_lr)')
        config.model.base_learning_rate = lr
    else:
        lr = base_lr
        print(f'Learning rate: {lr:.2e}')

    # ── instantiate model ─────────────────────────────────────────────────────
    model = instantiate_from_config(config.model)

    # ── instantiate data module ───────────────────────────────────────────────
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        SetupCallback(
            resume=opt.resume,
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=config,
            lightning_config=lightning_config,
        ),
        ModelCheckpoint(
            dirpath=ckptdir,
            filename='{epoch:06d}',
            verbose=True,
            save_last=True,
            every_n_epochs=10,
        ),
        LearningRateMonitor(logging_interval='step'),
        ImageLogger(
            batch_frequency=2000,
            max_images=4,
            clamp=True,
        ),
    ]

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(trainer_config)
    trainer_kwargs['max_epochs'] = opt.max_epochs
    trainer_kwargs['callbacks'] = callbacks
    trainer_kwargs['default_root_dir'] = logdir

    trainer = Trainer(**trainer_kwargs)

    # ── train ─────────────────────────────────────────────────────────────────
    try:
        trainer.fit(model, data, ckpt_path=ckpt_path)
    except Exception as e:
        # Save emergency checkpoint
        emergency_ckpt = os.path.join(ckptdir, 'emergency.ckpt')
        trainer.save_checkpoint(emergency_ckpt)
        raise e
