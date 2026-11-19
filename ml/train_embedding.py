"""Offline metric-learning projection training; the DINOv2 backbone stays frozen.

Run from the project root with the service PYTHONPATH, for example::

    python -m ml.train_embedding --manifest captures/embedding.json \
        --image-root captures --staging-root data/embedding-staging --version trial-001

Manifest JSON is {"samples": [...], "gallery": [...]}. Each sample has path,
card_id, source_group, split (train/val/test), kind (photo/reference), and photos
require session_id. Paths must stay inside --image-root. Train needs two classes
and two independently sourced, different images per class, or explicit
--synthetic-views 2 (or more) for reference-only classes. Augmented views retain
one source group. Validation must contain every trained class from disjoint
source groups, sessions and file contents, including separation from the gallery.
Gallery rows supply local reference paths and real card IDs, optionally metadata.
Test rows are decoded/hashed for leakage only: never embedded, scored or optimized here.
Validation is measured once after fixed epochs; no best-checkpoint selection.
Exports are a new staging directory, never activated and never written under
active data/models. Synthetic fixture results are not real-camera accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid

import faiss
import numpy as np
from PIL import ImageEnhance, ImageFilter, Image
import torch
from torch import nn
from torch.nn import functional as F

from cardlens_ai.identity import (PROJECTION_KIND, contained_path, identity_fingerprint, sha256)
from cardlens_ai.imaging import decode, phash
from cardlens_ai.pipeline import DATA, MODEL_NAME, PREPROCESS, backbone_embedding


def load_manifest(path, image_root, synthetic_views=0):
    if not isinstance(synthetic_views, int) or synthetic_views not in (0, *range(2, 17)):
        raise ValueError('synthetic_views_must_be_zero_or_2_to_16')
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    samples, gallery = raw.get('samples', []), raw.get('gallery', [])
    if not samples or not gallery:
        raise ValueError('manifest_requires_samples_and_gallery')
    by_split = {'train': [], 'val': [], 'test': [], 'gallery': []}
    groups, sessions, contents, content_labels = {}, {}, {}, {}
    for item in [*samples, *({**row, 'split': 'gallery'} for row in gallery)]:
        row = dict(item)
        split = row.get('split')
        if split not in by_split or (split == 'gallery' and item in samples):
            raise ValueError('invalid_manifest_split')
        if not isinstance(row.get('card_id'), str) or not row['card_id'].strip():
            raise ValueError('manifest_requires_card_id')
        if not isinstance(row.get('source_group'), str) or not row['source_group'].strip():
            raise ValueError('manifest_requires_source_group')
        if split != 'gallery' and row.get('kind') not in ('photo', 'reference'):
            raise ValueError('manifest_requires_photo_or_reference_kind')
        if row.get('kind') == 'photo' and not row.get('session_id'):
            raise ValueError('photo_requires_session_id')
        row['path'] = str(contained_path(image_root, row['path']))
        file = Path(row['path'])
        if not file.is_file() or not 0 < file.stat().st_size <= 8 * 1024 * 1024:
            raise ValueError('training_image_missing_or_size_limit')
        row['sha256'] = sha256(file)
        image = decode(file.read_bytes())
        row['pixel_sha256'] = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        label = content_labels.setdefault(row['pixel_sha256'], row['card_id'])
        if label != row['card_id']:
            raise ValueError('same_image_has_conflicting_labels')
        groups.setdefault(row['source_group'], set()).add(split)
        if row.get('session_id'):
            sessions.setdefault(row['session_id'], set()).add(split)
        contents.setdefault(row['pixel_sha256'], set()).add(split)
        by_split[split].append(row)
    for mapping, reason in ((groups, 'source_group_leakage'), (sessions, 'session_leakage'), (contents, 'content_leakage')):
        # Gallery is the retrieval reference and can share training sources only.
        if any(len(splits) > 1 and not splits <= {'train', 'gallery'} for splits in mapping.values()):
            raise ValueError(reason)
    train, val = by_split['train'], by_split['val']
    classes = {row['card_id'] for row in train}
    if len(classes) < 2 or not val:
        raise ValueError('requires_two_training_classes_and_validation')
    if not classes <= {row['card_id'] for row in val}:
        raise ValueError('validation_missing_training_class')
    if not classes <= {row['card_id'] for row in by_split['gallery']}:
        raise ValueError('gallery_missing_training_class')
    if len({row['card_id'] for row in by_split['gallery']}) != len(by_split['gallery']):
        raise ValueError('duplicate_gallery_card_id')
    for card_id in classes:
        rows = [row for row in train if row['card_id'] == card_id]
        independent = len({row['source_group'] for row in rows}) >= 2 and len({row['pixel_sha256'] for row in rows}) >= 2
        if not independent and not (synthetic_views >= 2 and all(row['kind'] == 'reference' for row in rows)):
            raise ValueError('training_requires_multiple_independent_samples_or_explicit_synthetic_views')
    return {**by_split, 'manifest_sha256': sha256(path), 'synthetic_views': synthetic_views,
            'data_kind': 'reference_augmentations_and_disjoint_validation' if synthetic_views else 'labeled_independent_samples'}


def _triplet_loss(head, features, labels, margin):
    projected = F.normalize(head(features), dim=1)
    distances = torch.cdist(projected, projected, p=2)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool)
    hardest_positive = distances.masked_fill(~positive, float('-inf')).max(dim=1).values
    hardest_negative = distances.masked_fill(same, float('inf')).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def fit_projection(features, labels, *, output_dim=128, epochs=50, learning_rate=.01, margin=.2, seed=17):
    """Actual full-batch hard-triplet optimization of a bias-free linear head on CPU.

    shortcut: cap at 4096 encoded views for bounded pairwise memory; use balanced
    minibatches when training a larger dataset. No validation/test inputs accepted.
    """
    features = torch.as_tensor(features, dtype=torch.float32).detach().cpu().clone()
    if (features.ndim != 2 or not 4 <= len(features) <= 4096 or len(labels) != len(features)
            or not torch.isfinite(features).all() or not (features.norm(dim=1) > 0).all()):
        raise ValueError('invalid_training_features')
    if (not isinstance(output_dim, int) or not 1 <= output_dim <= features.shape[1]
            or not isinstance(epochs, int) or not 1 <= epochs <= 5000
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1
            or not math.isfinite(margin) or not 0 < margin <= 2):
        raise ValueError('invalid_projection_hyperparameters')
    classes = sorted(set(labels))
    if len(classes) < 2 or any(labels.count(label) < 2 for label in classes):
        raise ValueError('triplet_loss_requires_two_classes_and_positives')
    targets = torch.tensor([classes.index(label) for label in labels])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(features.shape[1], output_dim, bias=False)
        initial_weight = head.weight.detach().clone()
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
        history = []
        for step in range(epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _triplet_loss(head, features, targets, margin)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite_training_loss')
            history.append(float(loss.detach()))
            if step < epochs:
                loss.backward()
                optimizer.step()
        if torch.equal(initial_weight, head.weight.detach()):
            raise ValueError('no_metric_learning_signal_projection_unchanged')
        return head.weight.detach().numpy().astype('float32', copy=True), history


def _augment(image, rng):
    # All controlled views keep their reference source_group; no split by augmentation.
    image = image.rotate(rng.uniform(-12, 12), resample=Image.Resampling.BICUBIC, fillcolor=(127, 127, 127))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.75, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.8, 1.2))
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0, .7)))


def extract_features(rows, *, synthetic_views=0, seed=17):
    vectors, labels = [], []
    rng = random.Random(seed)
    for row in rows:
        image = decode(Path(row['path']).read_bytes())
        views = synthetic_views if synthetic_views and row.get('kind') == 'reference' else 1
        for _ in range(views):
            view = _augment(image, rng) if views > 1 else image
            vectors.append(backbone_embedding(view)[0])
            labels.append(row['card_id'])
    return np.stack(vectors).astype('float32'), labels


def _project(features, weight):
    vectors = np.ascontiguousarray(features @ weight.T, dtype='float32')
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) <= 1e-8).any():
        raise ValueError('degenerate_projection')
    faiss.normalize_L2(vectors)
    return vectors


def export_version(staging_root, version, weight, vectors, hashes, rows, backbone_hashes, training, *, active_models=None):
    """Export inert array weights plus matching FAISS/hash/metadata in a new directory."""
    root = Path(staging_root).resolve()
    active = Path(active_models or DATA / 'models').resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', version):
        raise ValueError('invalid_embedding_version')
    target = contained_path(root, version)
    if root.is_relative_to(active) or active.is_relative_to(target):
        raise ValueError('staging_must_not_be_active_models')
    if target.exists():
        raise ValueError('version_already_exists')
    weight = np.asarray(weight)
    vectors = np.array(vectors, dtype='float32', order='C', copy=True)
    hashes = np.asarray(hashes)
    if (weight.ndim != 2 or weight.dtype != np.float32 or not np.isfinite(weight).all()
            or not 1 <= weight.shape[0] <= weight.shape[1] <= 4096
            or vectors.shape != (len(rows), weight.shape[0]) or not len(rows)
            or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) <= 1e-8).any()
            or hashes.shape != (len(rows), 63) or hashes.dtype != np.uint8
            or not np.isin(hashes, [0, 1]).all() or len({row['card_id'] for row in rows}) != len(rows)):
        raise ValueError('invalid_embedding_export_arrays')
    if set(backbone_hashes) != {'config.json', 'model.safetensors'} or any(not re.fullmatch(r'[a-f0-9]{64}', value) for value in backbone_hashes.values()):
        raise ValueError('invalid_backbone_checksums')
    root.mkdir(parents=True, exist_ok=True)
    pending = root / ('.' + version + '-' + uuid.uuid4().hex)
    pending.mkdir()
    try:
        np.savez(pending / 'projection.npz', weight=weight)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(weight.shape[0])
        index.add(vectors)
        faiss.write_index(index, str(pending / 'gallery.faiss'))
        np.save(pending / 'gallery-phash.npy', hashes, allow_pickle=False)
        projection = {'file': 'projection.npz', 'sha256': sha256(pending / 'projection.npz'),
                      'input_dim': weight.shape[1], 'output_dim': weight.shape[0]}
        registry = {'model': MODEL_NAME, 'checkpoint_kind': PROJECTION_KIND, 'backbone_trained': False,
                    'preprocessing': PREPROCESS, 'embedding_sha256': backbone_hashes, 'projection': projection,
                    'index_sha256': sha256(pending / 'gallery.faiss'),
                    'phash_sha256': sha256(pending / 'gallery-phash.npy'), 'count': len(rows), 'rows': rows,
                    'gallery_version': hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                    'version': version, 'activation': 'staging_only_not_activated', 'calibration': 'not_calibrated',
                    'real_photo_validated': False, 'pose': 'not_available', 'roi': 'fixed_bottom_header_v3',
                    'training': training}
        registry['identity_fingerprint'] = identity_fingerprint(registry)
        (pending / 'registry.json').write_text(json.dumps(registry, indent=2, allow_nan=False), encoding='utf-8')
        if target.exists():
            raise ValueError('version_already_exists')
        pending.rename(target)
    except Exception:
        shutil.rmtree(pending)
        raise
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--staging-root', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--synthetic-views', type=int, default=0)
    parser.add_argument('--output-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args(argv)
    active = (DATA / 'models').resolve()
    root = args.staging_root.resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.version):
        raise ValueError('invalid_embedding_version')
    target = contained_path(root, args.version)
    if root.is_relative_to(active) or active.is_relative_to(target):
        raise ValueError('staging_must_not_be_active_models')
    if target.exists():
        raise ValueError('version_already_exists')
    data = load_manifest(args.manifest, args.image_root, args.synthetic_views)
    count = sum(args.synthetic_views if args.synthetic_views and row['kind'] == 'reference' else 1 for row in data['train'])
    if count > 4096:
        raise ValueError('training_view_limit_4096')
    features, labels = extract_features(data['train'], synthetic_views=args.synthetic_views, seed=args.seed)
    weight, history = fit_projection(features, labels, output_dim=args.output_dim, epochs=args.epochs,
                                     learning_rate=args.learning_rate, margin=args.margin, seed=args.seed)
    gallery_features, gallery_labels = extract_features(data['gallery'])
    vectors = _project(gallery_features, weight)
    # Validation occurs once after optimization and cannot choose epoch/thresholds.
    validation, validation_labels = extract_features(data['val'])
    scores = _project(validation, weight) @ vectors.T
    correct = sum(gallery_labels[int(i)] == label for i, label in zip(scores.argmax(1), validation_labels))
    hashes = np.stack([phash(decode(Path(row['path']).read_bytes())) for row in data['gallery']])
    from cardlens_ai.identity import catalog_metadata
    rows = [catalog_metadata(row, DATA) if row.get('provider_id') else
            {key: row.get(key) for key in ('card_id', 'name', 'set', 'set_id', 'set_codes', 'number', 'official_total', 'language')}
            for row in data['gallery']]
    for row in rows:
        row['variant'] = None
    backbone = active / 'dinov2-small'
    checksums = json.loads((backbone / 'checksums.json').read_text(encoding='utf-8'))
    checksums = {key: checksums[key] for key in ('config.json', 'model.safetensors')}
    if any(sha256(backbone / key) != value for key, value in checksums.items()):
        raise ValueError('embedding_checkpoint_checksum_mismatch')
    training = {'objective': 'batch_hard_triplet', 'optimizer': 'Adam', 'epochs': args.epochs,
                'learning_rate': args.learning_rate, 'margin': args.margin, 'seed': args.seed,
                'manifest_sha256': data['manifest_sha256'], 'data_kind': data['data_kind'],
                'train_view_count': len(labels), 'synthetic_views_per_reference': args.synthetic_views,
                'augmentations': 'rotation_12deg_brightness_contrast_blur_v1' if args.synthetic_views else None,
                'train_source_groups': sorted({row['source_group'] for row in data['train']}),
                'validation_source_groups': sorted({row['source_group'] for row in data['val']}),
                'source_fingerprints': [{key: row.get(key) for key in ('card_id', 'sha256', 'pixel_sha256', 'source_group', 'session_id', 'kind', 'split')}
                                        for row in data['train'] + data['val'] + data['gallery']],
                'training_loss': history, 'selection': 'fixed_epochs_no_validation_selection',
                'validation': {'kind': 'provided_disjoint_validation_not_camera_accuracy', 'count': len(validation_labels),
                               'retrieval_top1': correct / len(validation_labels)},
                'test_optimized': False, 'test_evaluated': False, 'ignored_test_count': len(data['test'])}
    registry = export_version(args.staging_root, args.version, weight, vectors, hashes, rows, checksums, training)
    print(json.dumps({'directory': str(args.staging_root.resolve() / args.version),
                      'checkpoint_kind': registry['checkpoint_kind'], 'backbone_trained': False,
                      'identity_fingerprint': registry['identity_fingerprint'],
                      'validation': training['validation'], 'activation': registry['activation']}, indent=2))
    return registry


if __name__ == '__main__':
    main()
