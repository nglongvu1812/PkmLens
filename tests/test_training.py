import json
from pathlib import Path
import pytest
from PIL import Image
from ml.detection_dataset import generate


def test_yolo_import_normalizes_a_standard_dataset_without_editing_source(tmp_path):
    from ml.import_yolo import import_dataset
    source = tmp_path / 'source'
    for i, split in enumerate(('train', 'val', 'test')):
        (source / split / 'images').mkdir(parents=True)
        (source / split / 'labels').mkdir()
        Image.new('RGB', (100, 150), (50 + i * 50, 70, 90)).save(source / split / 'images/card.jpg')
        (source / split / 'labels/card.txt').write_text('0 0.5 0.5 0.8 0.8')
    yaml = source / 'data.yaml'
    yaml.write_text('train: train/images\nval: val/images\ntest: test/images\nnames: [card]\n')
    before = yaml.read_bytes()
    output = tmp_path / 'imported'
    result = import_dataset(yaml, output)
    assert result['valid']
    assert result['counts']['train']['images'] == 1
    assert yaml.read_bytes() == before
    assert json.loads((output / 'dataset.json').read_text())['kind'] == 'imported_unverified'


def test_import_rejects_paths_outside_selected_root_and_multiple_classes(tmp_path):
    from ml.import_yolo import import_dataset
    source = tmp_path / 'source'
    source.mkdir()
    config = source / 'data.yaml'
    config.write_text('train: ../private\nval: ../private\nnames: [card]\n')
    with pytest.raises(ValueError, match='outside'):
        import_dataset(config, tmp_path / 'output')
    config.write_text('train: images/train\nval: images/val\nnames: [card, person]\n')
    with pytest.raises(ValueError, match='one_class'):
        import_dataset(config, tmp_path / 'output2')


def test_training_resume_keeps_command_flags_and_uses_existing_output(tmp_path, monkeypatch):
    from cardlens_ai import training
    gallery = tmp_path / 'gallery'
    gallery.mkdir()
    for i in range(12):
        Image.new('RGB', (90, 130), (i * 15, 60, 100)).save(gallery / f'card-{i}.webp')
    dataset = tmp_path / 'datasets/cards'
    generate(gallery, dataset, count=6, size=320)
    previous = tmp_path / 'training/previous'
    (previous / 'weights').mkdir(parents=True)
    (previous / 'weights/last.pt').write_bytes(b'checkpoint')
    commands = []
    class Process:
        def poll(self): return 0
    monkeypatch.setattr(training, 'DATA', tmp_path)
    monkeypatch.setattr(training, 'processes', {})
    monkeypatch.setattr(training.subprocess, 'Popen', lambda command, **kwargs: commands.append(command) or Process())
    result = training.start(training.TrainInput(dataset='cards', resume_run='previous', checkpoint='last.pt'))
    assert result['id'] == 'previous'
    assert commands[0][4] == str(previous)
    assert commands[0][5:7] == ['--model', 'n']
    assert commands[0][-2:] == ['--resume', str(previous / 'weights/last.pt')]


def test_header_ocr_preserves_a_new_name_without_catalog():
    from cardlens_ai.pipeline import parse_name
    assert parse_name(['BASIC', 'Newmon ex', 'HP 280']) == 'Newmon ex'
    assert parse_name(['STAGE 2', '330 HP']) is None
    assert parse_name(['TRAII', 'TBAINE', 'TRAING', 'W ENER', 'SUPPORTER', 'Pokémon Flute']) == 'Pokémon Flute'


def test_ocr_recovers_a_number_merged_with_set_code_and_keeps_gallery_prefixes():
    from cardlens_ai.pipeline import parse_numbers
    assert parse_numbers(['0BF223/197'])['number'] == '223'
    assert parse_numbers(['OBF223/197'])['number'] == '223'
    assert parse_numbers(['OB223/197'])['number'] == '223'
    assert parse_numbers(['OfF223/197'])['number'] == '223'
    assert parse_numbers(['MEWE199/165★★']) == {'number': '199', 'total': '165', 'status': 'read'}
    assert parse_numbers(['GG15/070'])['number'] == 'GG15'
    assert parse_numbers(['TG01/030'])['number'] == 'TG01'
    assert parse_numbers(['007/102'])['number'] == '007'


def test_saved_resume_checkpoint_keeps_optimizer_when_final_eval_strips_last(tmp_path):
    from types import SimpleNamespace
    from ml.train_detection import preserve_resume_checkpoint
    last = tmp_path / 'last.pt'
    last.write_bytes(b'full checkpoint including optimizer')
    preserve_resume_checkpoint(SimpleNamespace(last=last))
    last.write_bytes(b'inference checkpoint')
    assert (tmp_path / 'resume.pt').read_bytes() == b'full checkpoint including optimizer'
