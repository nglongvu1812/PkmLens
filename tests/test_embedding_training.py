"""Embedding unit fixtures; optimizer uses real torch, never a real-photo benchmark."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from ml.train_embedding import fit_projection, load_manifest, export_version
from cardlens_ai.pipeline import load_projection
from cardlens_ai.identity import PROJECTION_KIND


def manifest_fixture(tmp_path):
    samples, gallery = [], []
    for label in range(2):
        for index, split in enumerate(('train', 'train', 'val')):
            name = f'{label}-{index}.png'
            pixels = np.random.default_rng(label * 10 + index).integers(10, 230, (64, 90, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(tmp_path / name)
            samples.append({'sample_id': name, 'path': name, 'card_id': f's:{label}:en', 'source_group': name,
                            'session_id': f'session-{index}', 'kind': 'photo', 'split': split})
        gallery.append({**samples[-3], 'sample_id': f'gallery-{label}', 'split': 'gallery', 'name': f'Fixture {label}'})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'samples': samples, 'gallery': gallery}))
    return path


def test_manifest_requires_independent_positives_and_disjoint_sources(tmp_path):
    path = manifest_fixture(tmp_path)
    data = load_manifest(path, tmp_path)
    assert len(data['train']) == 4 and len(data['val']) == 2
    raw = json.loads(path.read_text())
    raw['samples'][1]['source_group'] = raw['samples'][0]['source_group']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='independent_samples'):
        load_manifest(path, tmp_path)
    raw['samples'][1]['source_group'] = 'train-distinct'
    raw['samples'][2]['source_group'] = raw['samples'][0]['source_group']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='source_group_leakage'):
        load_manifest(path, tmp_path)


def test_manifest_rejects_session_content_leakage_and_traversal(tmp_path):
    path = manifest_fixture(tmp_path)
    raw = json.loads(path.read_text())
    raw['samples'][2]['session_id'] = 'session-0'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='session_leakage'):
        load_manifest(path, tmp_path)
    raw['samples'][2]['session_id'] = 'session-2'
    raw['samples'][2]['path'] = raw['samples'][0]['path']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='content_leakage'):
        load_manifest(path, tmp_path)
    raw['samples'][2]['path'] = '../outside.png'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='path_outside_allowed_root'):
        load_manifest(path, tmp_path)


def test_reference_requires_explicit_synthetic_views(tmp_path):
    path = manifest_fixture(tmp_path)
    raw = json.loads(path.read_text())
    raw['samples'] = [row for index, row in enumerate(raw['samples']) if index not in (1, 4)]
    for row in raw['samples']:
        if row['split'] == 'train':
            row['kind'] = 'reference'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='independent_samples'):
        load_manifest(path, tmp_path)
    assert len(load_manifest(path, tmp_path, synthetic_views=2)['train']) == 2


def test_real_optimizer_changes_projection_and_lowers_training_fixture_loss():
    features = torch.tensor([[1., .9, 0, 0], [.9, 1., 0, 0], [1., 0, .9, 0], [.9, 0, 1., 0]])
    labels = ['one', 'one', 'two', 'two']
    weight, history = fit_projection(features, labels, output_dim=2, epochs=30, learning_rate=.03, margin=1.5, seed=4)
    assert weight.shape == (2, 4) and np.isfinite(weight).all()
    assert history[-1] < history[0]
    assert len(history) == 31


def test_export_is_staged_versioned_fingerprinted_and_array_only(tmp_path):
    staging = tmp_path / 'staging'
    rows = [{'card_id': 'a:1:en'}, {'card_id': 'a:2:en'}]
    weight = np.eye(2, 4, dtype=np.float32)
    vectors = np.eye(2, dtype=np.float32)
    hashes = np.zeros((2, 63), dtype=np.uint8)
    registry = export_version(staging, 'fixture-v1', weight, vectors, hashes, rows,
                              {'config.json': 'a' * 64, 'model.safetensors': 'b' * 64},
                              {'manifest_sha256': 'c' * 64, 'kind': 'unit_fixture', 'training_loss': [.5, .1]})
    target = staging / 'fixture-v1'
    assert registry['checkpoint_kind'] == PROJECTION_KIND
    assert registry['backbone_trained'] is False
    assert registry['calibration'] == 'not_calibrated'
    assert registry['projection']['sha256'] == hashlib.sha256((target / 'projection.npz').read_bytes()).hexdigest()
    assert np.array_equal(load_projection(str(target / 'projection.npz'), registry['projection']['sha256'], 4, 2), weight)
    import faiss
    index = faiss.read_index(str(target / 'gallery.faiss'))
    assert index.ntotal == 2 and index.d == 2
    with pytest.raises(ValueError, match='version_already_exists'):
        export_version(staging, 'fixture-v1', weight, vectors, hashes, rows, {}, {})
    with pytest.raises(ValueError, match='staging_must_not_be_active_models'):
        export_version(tmp_path / 'data/models/nested', 'fixture-v2', weight, vectors, hashes, rows, {}, {}, active_models=tmp_path / 'data/models')
    assert not (tmp_path / 'data/models').exists()


def test_projection_refuses_bad_checksum_and_object_array(tmp_path):
    path = tmp_path / 'projection.npz'
    np.savez(path, weight=np.array([{'bad': 'pickle'}], dtype=object))
    with pytest.raises(ValueError, match='checksum_mismatch'):
        load_projection(str(path), '0' * 64, 4, 2)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_projection(str(path), checksum, 4, 2)


def test_projection_checkpoint_path_must_stay_in_version_directory(tmp_path):
    from cardlens_ai.pipeline import projection_for
    registry = {'checkpoint_kind': PROJECTION_KIND, 'projection': {'file': '../outside.npz'}}
    with pytest.raises(ValueError, match='path_outside_allowed_root'):
        projection_for(registry, tmp_path)


def test_test_samples_never_enter_optimizer_inputs(tmp_path):
    path = manifest_fixture(tmp_path)
    raw = json.loads(path.read_text())
    Image.new('RGB', (64, 90), 'red').save(tmp_path / 'held-out-test.png')
    raw['samples'].append({'path': 'held-out-test.png', 'card_id': 's:0:en', 'source_group': 'test-only',
                           'session_id': 'test-only', 'kind': 'photo', 'split': 'test'})
    path.write_text(json.dumps(raw))
    data = load_manifest(path, tmp_path)
    assert len(data['test']) == 1
    assert all(row['split'] == 'train' for row in data['train'])
    assert all(row['source_group'] != 'test-only' for row in data['train'] + data['val'] + data['gallery'])


def test_synthetic_views_keep_reference_source_group_and_are_reproducible(tmp_path, monkeypatch):
    from ml import train_embedding
    path = manifest_fixture(tmp_path)
    row = load_manifest(path, tmp_path)['train'][0]
    row['kind'] = 'reference'
    # Pixel-statistic stub isolates augmentation/grouping; this is not model validation.
    monkeypatch.setattr(train_embedding, 'backbone_embedding', lambda image: np.array([[np.asarray(image).mean(), np.asarray(image).std()]], dtype='float32'))
    features, labels = train_embedding.extract_features([row], synthetic_views=3, seed=4)
    repeated, _ = train_embedding.extract_features([row], synthetic_views=3, seed=4)
    assert features.shape == (3, 2) and labels == [row['card_id']] * 3
    assert np.array_equal(features, repeated)
    assert not np.array_equal(features[0], features[1])
    assert row['source_group'] == '0-0.png'


def test_reencoded_validation_image_cannot_evade_content_leakage(tmp_path):
    path = manifest_fixture(tmp_path)
    raw = json.loads(path.read_text())
    with Image.open(tmp_path / raw['samples'][0]['path']) as image:
        image.save(tmp_path / 'renamed.png', compress_level=0)
    raw['samples'][2]['path'] = 'renamed.png'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='content_leakage'):
        load_manifest(path, tmp_path)


def test_no_learning_signal_cannot_be_exported_as_finetuned():
    features = torch.tensor([[1., 0], [1., 0], [-1., 0], [-1., 0]])
    with pytest.raises(ValueError, match='no_metric_learning_signal'):
        fit_projection(features, ['one', 'one', 'two', 'two'], output_dim=2, epochs=2)
