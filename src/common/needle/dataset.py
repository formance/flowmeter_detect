"""Needle crop splits and polygon annotations."""
import json

import numpy as np

from src.common.needle.normalization import RGB_MEAN, RGB_STD


NEEDLE_LABELS = {'gauge needle', 'pointer'}


def load_splits(dataset, split_images):
    crops = {image.stem: image for image in dataset.glob('*.png')}
    if not crops:
        raise ValueError(f'No PNG crops found in {dataset}')
    assigned = set()
    splits = {}
    for split in ('train', 'val', 'test'):
        image_dir = split_images / split
        members = {path.stem for path in image_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
        stems = members & crops.keys()
        if not stems or assigned & stems:
            raise ValueError(f'{split}: expected a nonempty split with no shared crops')
        assigned.update(stems)
        splits[split] = [crops[stem] for stem in sorted(stems)]
    if missing := crops.keys() - assigned:
        raise ValueError(f'Crops missing from the existing split: {", ".join(sorted(missing))}')
    return splits


def read_polygons(annotation_path, image_name):
    annotation = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    if annotation['imagePath'] != image_name:
        raise ValueError(f'Annotation does not match {image_name}')
    width, height = annotation['imageWidth'], annotation['imageHeight']
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError(f'Invalid image dimensions in {annotation_path}')

    polygons = []
    for shape in annotation['shapes']:
        if shape['label'].strip().lower() not in NEEDLE_LABELS:
            continue
        points = np.asarray(shape['points'], dtype=np.float64)
        if (shape['shape_type'] != 'polygon' or points.ndim != 2 or points.shape[1] != 2
                or len(points) < 3 or not np.isfinite(points).all()
                or np.any(points < 0) or np.any(points > (width, height))):
            raise ValueError(f'Invalid needle polygon in {annotation_path}')
        contour = np.clip(np.rint(points), 0, (width - 1, height - 1)).astype(np.int32)
        if len(np.unique(contour, axis=0)) < 3:
            raise ValueError(f'Degenerate needle polygon in {annotation_path}')
        polygons.append(contour)
    if not polygons:
        raise ValueError(f'No needle polygons in {annotation_path}')
    return (height, width), polygons


class NeedleDataset:
    def __init__(self, images, annotations, size, augment=False):
        import cv2

        self.samples = []
        self.augment = augment
        for path in images:
            dimensions, polygons = read_polygons(annotations / path.with_suffix('.json').name, path.name)
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != dimensions:
                raise ValueError(f'Image unreadable or dimensions differ from the annotation: {path}')
            mask = np.zeros(dimensions, dtype=np.uint8)
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], color=1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            if not mask.any():
                raise ValueError(f'Needle mask is empty at training resolution: {path}')
            self.samples.append((image, mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, mask = self.samples[index]
        image = image.astype(np.float32) / 255.0
        if self.augment:
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-0.05, 0.05)
            image = np.clip(image * gain + bias, 0, 1)
        image = (image - RGB_MEAN) / RGB_STD
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32), mask[None].astype(np.float32)
