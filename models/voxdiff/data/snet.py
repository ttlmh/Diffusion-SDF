"""
ShapeNet SDF + Text Dataset for Diffusion-SDF training.

Expected data directory structure:
    data/
        ShapeNet/
            sdf/
                <category_id>/        e.g. 03001627 (chair), 04379243 (table)
                    <model_id>/
                        pc_sdf_sample.h5    # float32 array (262144,) = 64^3 SDF values
            text/
                captions.json         # {model_id: [caption1, caption2, ...]}
                  -- OR --
                captions.tablechair.csv   # raw Text2Shape CSV (auto-parsed if json absent)
            train_models.json         # [model_id, ...]  (optional split files)
            val_models.json
            test_models.json

Text captions come from Text2Shape (chairs + tables only). Other categories
are trained unconditionally (empty caption string).

See preprocess/prepare_text2shape.py to convert Text2Shape CSV and generate
split files from your SDF directory.
"""

import csv
import os
import json
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Default ShapeNet category IDs (13 standard categories used in AutoSDF/Diffusion-SDF)
# Note: Text2Shape only provides captions for 'chair' and 'table'.
SHAPENET_CAT_IDS = {
    'airplane': '02691156',
    'bench':    '02828884',
    'cabinet':  '02933112',
    'car':      '02958343',
    'chair':    '03001627',
    'monitor':  '03211117',
    'lamp':     '03636649',
    'speaker':  '03691459',
    'gun':      '04090263',
    'sofa':     '04256520',
    'table':    '04379243',
    'telephone':'04401088',
    'vessel':   '04530566',
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
    sdf = torch.from_numpy(sdf).reshape(1, 64, 64, 64)
    return sdf


def _load_captions(data_root: str) -> dict:
    """
    Load text captions for ShapeNet models.

    Priority:
    1. data_root/text/captions.json  — dict {model_id: [caption, ...]}
    2. data_root/text/captions.tablechair.csv  — Text2Shape raw CSV
       (download: wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv)

    Returns {model_id: [caption, ...]} or {} if neither file exists.
    """
    text_dir = os.path.join(data_root, 'text')

    json_path = os.path.join(text_dir, 'captions.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    csv_path = os.path.join(text_dir, 'captions.tablechair.csv')
    if os.path.exists(csv_path):
        captions = {}
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row.get('modelId') or row.get('model_id', '')
                description = row.get('description', '').strip()
                if model_id and description:
                    captions.setdefault(model_id, []).append(description)
        return captions

    return {}


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

        # Load captions — prefer captions.json, fall back to Text2Shape CSV
        self.captions = _load_captions(data_root)

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
