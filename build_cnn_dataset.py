"""Maintainer-only script: build the fixed teaching dataset once, outside Colab.

Run: python build_cnn_dataset.py
Students download the resulting data/icons_cnn_v2.npz instead of running this.
"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORDS = ['heart', 'face', 'robot', 'tree', 'spaceship']

def main():
    # Use the improved icon templates already provided in this repository.
    # Temporary PNGs are only used while preparing the published dataset.
    with tempfile.TemporaryDirectory() as temporary:
        subprocess.run([sys.executable, str(ROOT / 'generate_dataset_v2.py')],
                       cwd=temporary, check=True)
        source = Path(temporary) / 'pixel_dataset_v2'
        with (source / 'labels.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        images = []
        for row in rows:
            with Image.open(source / row['filename']) as image:
                images.append(np.asarray(image.convert('L'), dtype=np.uint8))
        images = np.stack(images)
        labels = np.array([int(row['label']) for row in rows], dtype=np.int32)

    assert images.shape == (10000, 16, 16)
    assert np.isin(images, [0, 255]).all()
    assert np.array_equal(np.bincount(labels, minlength=5), [2000] * 5)
    assert all(row['word'] == WORDS[int(row['label'])] for row in rows)

    destination = ROOT / 'data'
    destination.mkdir(exist_ok=True)
    archive = destination / 'icons_cnn_v2.npz'
    # Integer pixels keep the download small. Notebooks normalize to 0..1.
    np.savez_compressed(archive, images=images, labels=labels,
                        words=np.array(WORDS))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        'file': archive.name, 'sha256': checksum,
        'image_count': len(images), 'image_shape': [16, 16],
        'pixel_values': [0, 255], 'words': WORDS,
        'examples_per_word': 2000, 'source': 'generate_dataset_v2.py',
        'seed': '200000 + image index',
    }
    (destination / 'manifest_cnn_v2.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {archive.name}: {archive.stat().st_size:,} bytes')
    print(f'SHA256: {checksum}')

if __name__ == '__main__':
    main()
