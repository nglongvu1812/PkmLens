"""Reproducible multi-card scenes. Synthetic metrics are not real-camera accuracy."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def source_group(path):
    # This known reprint shares artwork; neither printing may leak across splits.
    return "hitmonchan-shared" if path.stem in ("base1-7", "base4-8") else path.stem


def split_for(path):
    bucket = int(hashlib.sha256(source_group(path).encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def artwork_groups(gallery):
    groups = {}
    for path in Path(gallery).glob('*.webp'):
        group = source_group(path)
        metadata = Path(gallery).parent / 'catalog' / (path.stem + '.json')
        if metadata.is_file():
            detail = json.loads(metadata.read_text(encoding='utf-8')).get('detail', {})
            painting = detail.get('original', {}).get('links', {}).get('painting')
            if isinstance(painting, dict) and painting.get('set') and painting.get('painting'):
                group = 'painting:' + painting['set'] + '/' + painting['painting']
        groups[path.stem] = group
    return groups


def generate(gallery, output, count=600, seed=26, size=640):
    output, gallery = Path(output), Path(gallery)
    if output.exists():
        raise ValueError("dataset_exists_choose_new_name")
    sources = sorted(gallery.glob("*.webp"))
    source_groups = artwork_groups(gallery)
    groups = sorted(set(source_groups.values()), key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError("need_at_least_three_source_groups")
    holdout = max(1, len(groups) // 10)
    assigned = {g: "test" if i < holdout else "val" if i < 2 * holdout else "train" for i, g in enumerate(groups)}
    pools = {s: [p for p in sources if assigned[source_groups[p.stem]] == s] for s in ("train", "val", "test")}
    if any(not p for p in pools.values()):
        raise ValueError("need_source_images_in_all_splits")
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    manifest = []
    for split, total in (("train", count), ("val", max(40, count // 6)), ("test", max(40, count // 6))):
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(total):
            base = np.array([rng.randint(40, 210) for _ in range(3)])
            canvas = np.clip(base + noise_rng.normal(0, rng.uniform(2, 15), (size, size, 3)), 0, 255).astype(np.uint8)
            if i % 3 == 0:
                for y in range(0, size, rng.randint(30, 70)):
                    cv2.line(canvas, (0, y), (size, y + rng.randint(-30, 30)), tuple(int(v) for v in base * .7), 2)
            # Rectangle distractors are negatives, never relabeled as cards.
            for _ in range(rng.randint(0, 3)):
                x, y = rng.randrange(size - 150), rng.randrange(size - 150)
                cv2.rectangle(canvas, (x, y), (x + rng.randint(50, 180), y + rng.randint(40, 180)),
                              (rng.randrange(255), rng.randrange(255), rng.randrange(255)), -1)
            amount = 0 if i % 7 == 0 else rng.randint(1, 4)
            objects, annotations = [], []
            cells = rng.sample([(size * .25, size * .25), (size * .75, size * .25),
                                (size * .75, size * .75), (size * .25, size * .75)], amount)
            for cx, cy in cells:
                source = rng.choice(pools[split])
                image = np.array(Image.open(source).convert("RGB").resize((252, 352)))
                height = rng.uniform(size * .25, size * (.65 if amount == 1 else .42))
                width = height * 63 / 88
                angle = rng.uniform(-np.pi, np.pi)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                quad = np.array([[-width/2, -height/2], [width/2, -height/2],
                                 [width/2, height/2], [-width/2, height/2]]) @ rotation.T
                quad += np.array([cx, cy]) + noise_rng.uniform(-10, 10, (4, 2))
                quad = np.clip(quad, 2, size - 3).astype(np.float32)
                if not cv2.isContourConvex(quad):
                    continue
                matrix = cv2.getPerspectiveTransform(np.float32([[0,0],[251,0],[251,351],[0,351]]), quad)
                warped = cv2.warpPerspective(image, matrix, (size, size))
                mask = cv2.warpPerspective(np.full((352,252),255,np.uint8), matrix, (size,size))
                canvas[mask > 0] = warped[mask > 0]
                if rng.random() < .2:
                    glare = np.zeros_like(canvas)
                    cv2.ellipse(glare, tuple(quad.mean(0).astype(int)), (int(width*.2), int(height*.1)), 30, 0, 360, (255,255,255), -1)
                    canvas = cv2.addWeighted(canvas, 1, glare, .35, 0)
                lo, hi = quad.min(0) / size, quad.max(0) / size
                annotations.append(" ".join(map(str, [0, *((lo+hi)/2), *(hi-lo)])))
                objects.append({"source": source.name, "group": source_groups[source.stem], "corners": quad.tolist()})
            if rng.random() < .3:
                canvas = cv2.GaussianBlur(canvas, (3,3), rng.uniform(.2, 1.2))
            name = f"{split}-{i:05d}"
            target = images / (name + ".jpg")
            Image.fromarray(canvas).save(target, quality=rng.randint(78, 96))
            (labels / (name + ".txt")).write_text("\n".join(annotations), encoding="utf-8")
            manifest.append({"sample_id": name, "split": split, "source_kind": "synthetic_catalog_scene",
                             "objects": objects, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (output / "dataset.yaml").write_text(json.dumps({"path": str(output.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test", "names": {0: "pokemon_card"}}), encoding="utf-8")
    (output / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest), encoding="utf-8")
    info = {"kind": "synthetic_only", "seed": seed, "count": len(manifest), "classes": ["pokemon_card"],
            "source_count": len(sources), "real_photo_count": 0,
            "sha256": hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest()}
    (output / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--gallery", default="data/gallery", type=Path)
    p.add_argument("--count", type=int, default=600)
    a = p.parse_args()
    print(generate(a.gallery, a.output, a.count))
