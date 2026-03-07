"""
ShapeNet SDF + Text Dataset for Diffusion-SDF training.

Expected data directory structure:
    data/
        ShapeNet/
            sdf/
                <category_id>/
                    <model_id>/
                        pc_sdf_sample.h5    # float32 SDF, shape (64,64,64)
            text/
                captions.json               # {model_id: [caption1, caption2, ...]}

The captions.json can be sourced from Text2Shape or ShapeGlot datasets.
"""

import os
import json
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Default ShapeNet category IDs (13 common categories)
SHAPENET_CAT_IDS = {
    'chair':    '03001627',
    'table':    '04379243',
    'airplane': '02691156',
    'car':      '02958343',
    'sofa':     '04256520',
    'lamp':     '03636649',
    'cabinet':  '02933112',
    'monitor':  '03211117',
    'gun':      '04090263',
    'couch':    '04401088',
    'vessel':   '04530566',
    'speaker':  '03691459',
    'telephone':'04401088',
}

ALL_CATS = list(SHAPENET_CAT_IDS.values())


def load_sdf(h5_path):
    """Load SDF from h5 file. Returns tensor (1, 64, 64, 64)."""
    with h5py.File(h5_path, 'r') as f:
        # Support different key names
        if 'pc_sdf_sample' in f:
            sdf = f['pc_sdf_sample'][:].astype(np.float32)
        elif 'sdf' in f:
            sdf = f['sdf'][:].astype(np.float32)
        else:
            key = list(f.keys())[0]
            sdf = f[key][:].astype(np.float32)
    sdf = torch.from_numpy(sdf).view(1, 64, 64, 64)
    return sdf


class ShapeNetTextDatasetBase(Dataset):
    """
    Base ShapeNet dataset that provides voxelized SDF and text captions.

    Args:
        data_root:  path to the data directory containing 'sdf/' and 'text/'
        split:      'train' or 'val'
        cat:        category name (from SHAPENET_CAT_IDS), or 'all'
        thres:      SDF clamp threshold (clip SDF to [-thres, thres])
        ucond_p:    probability of returning empty string (unconditional training)
        prompt:     if set, always use this text (useful for debugging)
    """

    def __init__(
        self,
        data_root: str = 'data/ShapeNet',
        split: str = 'train',
        cat: str = 'all',
        thres: float = 0.2,
        ucond_p: float = 0.0,
        prompt: str = '',
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.thres = thres
        self.ucond_p = ucond_p
        self.prompt = prompt

        # Determine categories
        if cat == 'all':
            self.cat_ids = ALL_CATS
        elif cat in SHAPENET_CAT_IDS:
            self.cat_ids = [SHAPENET_CAT_IDS[cat]]
        else:
            # Treat as raw category ID
            self.cat_ids = [cat]

        # Load captions
        caption_path = os.path.join(data_root, 'text', 'captions.json')
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                self.captions = json.load(f)
        else:
            # Fallback: no captions available, use empty strings
            self.captions = {}

        # Collect all SDF samples
        self.samples = self._collect_samples()

    def _collect_samples(self):
        """Collect (sdf_path, model_id) for all available models."""
        samples = []
        sdf_root = os.path.join(self.data_root, 'sdf')

        # Load train/val split if available
        split_file = os.path.join(self.data_root, f'{self.split}_models.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_ids = set(json.load(f))
        else:
            split_ids = None

        for cat_id in self.cat_ids:
            cat_dir = os.path.join(sdf_root, cat_id)
            if not os.path.isdir(cat_dir):
                continue

            for model_id in sorted(os.listdir(cat_dir)):
                # Apply split filtering
                if split_ids is not None:
                    if model_id not in split_ids:
                        continue

                sdf_path = os.path.join(cat_dir, model_id, 'pc_sdf_sample.h5')
                if not os.path.exists(sdf_path):
                    # Try alternative filenames
                    for fname in ['sdf.h5', 'model.h5']:
                        alt = os.path.join(cat_dir, model_id, fname)
                        if os.path.exists(alt):
                            sdf_path = alt
                            break
                    else:
                        continue

                samples.append((sdf_path, model_id))

        return samples

    def __len__(self):
        return len(self.samples)

    def _get_caption(self, model_id):
        """Return a text caption for the given model."""
        if self.prompt:
            return self.prompt

        if model_id in self.captions:
            caps = self.captions[model_id]
            if isinstance(caps, list) and len(caps) > 0:
                return random.choice(caps)
            return str(caps)

        return ''

    def __getitem__(self, idx):
        sdf_path, model_id = self.samples[idx]

        # Load and preprocess SDF
        sdf = load_sdf(sdf_path)  # (1, 64, 64, 64)

        # Clamp SDF to [-thres, thres] and normalize to roughly [-1, 1]
        if self.thres > 0:
            sdf = torch.clamp(sdf, -self.thres, self.thres)
            sdf = sdf / self.thres  # normalize to [-1, 1]

        # Get text caption
        caption = self._get_caption(model_id)

        # Randomly drop caption for unconditional training (classifier-free guidance)
        if self.ucond_p > 0 and random.random() < self.ucond_p:
            caption = ''

        return {
            'sdf': sdf,          # float32 tensor (1, 64, 64, 64)
            'caption': caption,  # str
            'model_id': model_id,
        }


class ShapeNetTextDatasetTrain(ShapeNetTextDatasetBase):
    """Training split of ShapeNet SDF + text dataset."""

    def __init__(self, **kwargs):
        kwargs.setdefault('split', 'train')
        super().__init__(**kwargs)


class ShapeNetTextDatasetValidation(ShapeNetTextDatasetBase):
    """Validation split of ShapeNet SDF + text dataset."""

    def __init__(self, **kwargs):
        kwargs.setdefault('split', 'val')
        kwargs.setdefault('ucond_p', 0.0)  # no dropout at validation
        super().__init__(**kwargs)


class ShapeNetSDFDataset(Dataset):
    """
    Dataset for AE training (SDF only, no text).

    Args:
        data_root:  path to the data directory containing 'sdf/'
        split:      'train' or 'val'
        cat:        category name or 'all'
        thres:      SDF clamp threshold
    """

    def __init__(
        self,
        data_root: str = 'data/ShapeNet',
        split: str = 'train',
        cat: str = 'all',
        thres: float = 0.2,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.thres = thres

        if cat == 'all':
            self.cat_ids = ALL_CATS
        elif cat in SHAPENET_CAT_IDS:
            self.cat_ids = [SHAPENET_CAT_IDS[cat]]
        else:
            self.cat_ids = [cat]

        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        sdf_root = os.path.join(self.data_root, 'sdf')

        split_file = os.path.join(self.data_root, f'{self.split}_models.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_ids = set(json.load(f))
        else:
            split_ids = None

        for cat_id in self.cat_ids:
            cat_dir = os.path.join(sdf_root, cat_id)
            if not os.path.isdir(cat_dir):
                continue

            for model_id in sorted(os.listdir(cat_dir)):
                if split_ids is not None and model_id not in split_ids:
                    continue

                sdf_path = os.path.join(cat_dir, model_id, 'pc_sdf_sample.h5')
                if not os.path.exists(sdf_path):
                    for fname in ['sdf.h5', 'model.h5']:
                        alt = os.path.join(cat_dir, model_id, fname)
                        if os.path.exists(alt):
                            sdf_path = alt
                            break
                    else:
                        continue

                samples.append(sdf_path)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sdf = load_sdf(self.samples[idx])  # (1, 64, 64, 64)

        if self.thres > 0:
            sdf = torch.clamp(sdf, -self.thres, self.thres)
            sdf = sdf / self.thres

        return {'sdf': sdf}
