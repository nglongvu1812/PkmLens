"""Copy a selected, single-class YOLO dataset into an independent validated dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import yaml


def import_dataset(config_file, output):
    config_file, output = Path(config_file).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('dataset_exists_choose_new_name')
    config = yaml.safe_load(config_file.read_text(encoding='utf-8-sig'))
    names = config.get('names') if isinstance(config, dict) else None
    if not isinstance(names, (list, dict)) or len(names) != 1 or (isinstance(names, dict) and set(map(str, names)) != {'0'}):
        raise ValueError('dataset_must_have_one_class')
    selected_root = config_file.parent
    root = (selected_root / config.get('path', '.')).resolve()
    if not root.is_relative_to(selected_root):
        raise ValueError('dataset_path_outside_selected_root')
    splits = {}
    for split in ('train', 'val', 'test'):
        value = config.get(split)
        if not value and split == 'test':
            continue
        if not isinstance(value, str):
            raise ValueError('dataset_split_directory_required:' + split)
        path = (root / value).resolve()
        if not path.is_relative_to(selected_root):
            raise ValueError('dataset_split_outside_selected_root')
        if not path.is_dir() or path.name != 'images':
            # Both images/train and train/images are standard layouts.
            if not path.is_dir() or 'images' not in path.parts:
                raise ValueError('dataset_images_directory_missing:' + split)
        splits[split] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pkmlens-import-', dir=output.parent) as temp:
        staged = Path(temp) / 'dataset'
        manifest = []
        for split, path in splits.items():
            images, labels = staged / 'images' / split, staged / 'labels' / split
            images.mkdir(parents=True); labels.mkdir(parents=True)
            for index, source in enumerate(sorted(path.rglob('*'))):
                if source.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
                    continue
                if not source.resolve().is_relative_to(selected_root):
                    raise ValueError('dataset_image_outside_selected_root')
                parts = list(source.relative_to(root).parts)
                image_segment = len(parts) - 1 - list(reversed(parts)).index('images')
                parts[image_segment] = 'labels'
                label = (root / Path(*parts)).with_suffix('.txt')
                if not label.resolve().is_relative_to(selected_root) or not label.is_file():
                    raise ValueError('dataset_label_missing:' + source.name)
                name = f'{split}-{index:06d}'
                shutil.copy2(source, images / (name + source.suffix.lower()))
                shutil.copy2(label, labels / (name + '.txt'))
                manifest.append({'sample_id': name, 'split': split, 'source_kind': 'imported_unverified',
                                 'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        (staged / 'manifest.jsonl').write_text('\n'.join(json.dumps(row) for row in manifest), encoding='utf-8')
        normalized = {'path': str(staged), 'names': {0: 'pokemon_card'}, **{split: 'images/' + split for split in splits}}
        (staged / 'dataset.yaml').write_text(yaml.safe_dump(normalized), encoding='utf-8')
        from cardlens_ai.training import validate_dataset
        validation = validate_dataset(staged)
        if not validation['valid']:
            raise ValueError('dataset_validation_failed:' + ';'.join(validation['errors']))
        info = {'kind': 'imported_unverified', 'classes': ['pokemon_card'], 'count': len(manifest),
                'real_photo_count': 0, 'sha256': validation['manifest_sha256'], 'source_config': str(config_file),
                'note': 'Imported labels and real-photo provenance require independent review.'}
        (staged / 'dataset.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        normalized['path'] = str(output)
        (staged / 'dataset.yaml').write_text(yaml.safe_dump(normalized), encoding='utf-8')
        (staged / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
        staged.rename(output)
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(import_dataset(args.config, args.output), indent=2))
