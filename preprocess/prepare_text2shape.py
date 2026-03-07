"""
Prepare Text2Shape captions and train/val/test split files for Diffusion-SDF.

Text2Shape (Chen et al., ACCV 2018) provides natural-language descriptions for
ShapeNet chairs (03001627) and tables (04379243) only.

Steps
-----
1. Download the caption CSV:
       wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv
   Place it at:  data/ShapeNet/text/captions.tablechair.csv

2. Download ShapeNet official train/val/test splits (v1 or v2) and place at:
       data/ShapeNet/splits/<synset_id>.{train,val,test}.json
   (Each file is a list of model IDs.)
   Alternatively, pass --shapenet_split_dir to point at your splits directory.

3. Run this script:
       python preprocess/prepare_text2shape.py --data_root data/ShapeNet

Outputs
-------
  data/ShapeNet/text/captions.json        {model_id: [caption, ...]}
  data/ShapeNet/train_models.json         [model_id, ...]
  data/ShapeNet/val_models.json           [model_id, ...]
  data/ShapeNet/test_models.json          [model_id, ...]

The split JSON files contain model IDs from *all* categories (not just chairs/
tables), so they can be used with --cat all for AE training even though only
chairs and tables will have text captions.

Notes
-----
- Text2Shape covers ONLY chairs and tables. Other ShapeNet categories will
  train unconditionally (empty caption) during diffusion model training.
- ShapeNet official splits are version-specific (v1 vs v2).  Make sure the
  model IDs in the splits match those in your SDF directory.
"""

import argparse
import csv
import json
import os
import random


# ShapeNet 13-class synset IDs (same as SHAPENET_CAT_IDS in snet.py)
ALL_CAT_IDS = [
    '02691156',  # airplane
    '02828884',  # bench
    '02933112',  # cabinet
    '02958343',  # car
    '03001627',  # chair
    '03211117',  # monitor
    '03636649',  # lamp
    '03691459',  # speaker
    '04090263',  # gun/rifle
    '04256520',  # sofa
    '04379243',  # table
    '04401088',  # telephone
    '04530566',  # vessel
]


def parse_args():
    p = argparse.ArgumentParser(description='Prepare Text2Shape data for Diffusion-SDF')
    p.add_argument('--data_root', type=str, default='data/ShapeNet',
                   help='Root data directory (contains sdf/ and text/)')
    p.add_argument('--csv', type=str, default='',
                   help='Path to captions.tablechair.csv '
                        '(default: <data_root>/text/captions.tablechair.csv)')
    p.add_argument('--shapenet_split_dir', type=str, default='',
                   help='Directory with ShapeNet split JSON files. '
                        'If not given, a random 85/10/5 split is created '
                        'from the models found in <data_root>/sdf/.')
    p.add_argument('--val_ratio', type=float, default=0.10,
                   help='Validation fraction when generating automatic splits.')
    p.add_argument('--test_ratio', type=float, default=0.05,
                   help='Test fraction when generating automatic splits.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Caption conversion
# ---------------------------------------------------------------------------

def build_captions_json(csv_path: str) -> dict:
    """Convert captions.tablechair.csv → {model_id: [caption, ...]}."""
    captions = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = (row.get('modelId') or row.get('model_id', '')).strip()
            description = row.get('description', '').strip()
            if model_id and description:
                captions.setdefault(model_id, []).append(description)
    return captions


# ---------------------------------------------------------------------------
# Split files
# ---------------------------------------------------------------------------

def collect_model_ids(sdf_root: str) -> list:
    """Walk sdf/<cat_id>/<model_id>/ and return all model IDs found."""
    model_ids = []
    for cat_id in ALL_CAT_IDS:
        cat_dir = os.path.join(sdf_root, cat_id)
        if not os.path.isdir(cat_dir):
            continue
        for model_id in os.listdir(cat_dir):
            if os.path.isdir(os.path.join(cat_dir, model_id)):
                model_ids.append(model_id)
    return model_ids


def make_random_splits(model_ids: list, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    ids = list(model_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = ids[:n_test]
    val = ids[n_test: n_test + n_val]
    train = ids[n_test + n_val:]
    return train, val, test


def load_shapenet_splits(split_dir: str) -> tuple:
    """
    Load ShapeNet official splits from a directory containing files like:
        <synset_id>.train.json   (list of model IDs)
        <synset_id>.val.json
        <synset_id>.test.json
    Returns (train_ids, val_ids, test_ids) as sets.
    """
    train_ids, val_ids, test_ids = set(), set(), set()
    for split, target in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        for cat_id in ALL_CAT_IDS:
            path = os.path.join(split_dir, f'{cat_id}.{split}.json')
            if os.path.exists(path):
                with open(path) as f:
                    target.update(json.load(f))
    return list(train_ids), list(val_ids), list(test_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    text_dir = os.path.join(args.data_root, 'text')
    sdf_dir = os.path.join(args.data_root, 'sdf')
    os.makedirs(text_dir, exist_ok=True)

    # ── Captions ────────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(text_dir, 'captions.tablechair.csv')
    if not os.path.exists(csv_path):
        print(f'[!] Caption CSV not found: {csv_path}')
        print('    Download it with:')
        print('      wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv'
              f' -P {text_dir}')
    else:
        captions = build_captions_json(csv_path)
        out_json = os.path.join(text_dir, 'captions.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)
        print(f'[+] Wrote {len(captions)} model captions → {out_json}')

    # ── Splits ──────────────────────────────────────────────────────────────
    if args.shapenet_split_dir and os.path.isdir(args.shapenet_split_dir):
        train, val, test = load_shapenet_splits(args.shapenet_split_dir)
        print(f'[+] Loaded ShapeNet splits: {len(train)} train / {len(val)} val / {len(test)} test')
    else:
        if not os.path.isdir(sdf_dir):
            print(f'[!] SDF directory not found: {sdf_dir}. Cannot generate splits.')
            return
        all_ids = collect_model_ids(sdf_dir)
        if not all_ids:
            print(f'[!] No model directories found in {sdf_dir}.')
            return
        train, val, test = make_random_splits(all_ids, args.val_ratio, args.test_ratio, args.seed)
        print(f'[+] Generated random splits from {len(all_ids)} models: '
              f'{len(train)} train / {len(val)} val / {len(test)} test')

    for name, ids in [('train', train), ('val', val), ('test', test)]:
        out_path = os.path.join(args.data_root, f'{name}_models.json')
        with open(out_path, 'w') as f:
            json.dump(sorted(ids), f, indent=2)
        print(f'[+] Wrote {name} split ({len(ids)} models) → {out_path}')

    print('[+] Done.')


if __name__ == '__main__':
    main()
