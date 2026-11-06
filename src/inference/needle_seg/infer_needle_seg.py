"""Run the trained needle segmenter on the held-out crops and save overlays."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.image_input import load_image
from src.common.paths import PROJECT_ROOT
from src.common.visualization import blend_labels, save_png
from src.common.needle.dataset import load_splits
from src.common.needle.normalization import RGB_MEAN, RGB_STD


def predict_mask(model, crop, size):
    import cv2
    import torch

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    normalized = (rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]))
    mask = (model(tensor)[0, 0] >= 0).numpy().astype(np.uint8)
    height, width = crop.shape[:2]
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path, help='Needle segmentation best.pt')
    parser.add_argument('--dataset', type=Path, default=PROJECT_ROOT / 'data/scene01_seg_min190')
    parser.add_argument('--split-images', type=Path, default=PROJECT_ROOT / 'data/scene01_yolo/images')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    if not args.weights.is_file() or args.weights.suffix.lower() != '.pt':
        parser.error(f'Expected an existing .pt checkpoint: {args.weights}')
    if args.threads < 1:
        parser.error('--threads must be positive')
    try:
        images = load_splits(args.dataset, args.split_images)['test']
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import cv2
    import torch
    from src.common.needle.needle_seg_model import NeedleDeepLabV3Plus

    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('architecture') != 'deeplabv3plus_resnet18':
        parser.error('--weights must be a checkpoint from train_needle_seg.py')
    size = checkpoint.get('imgsz')
    if type(size) is not int or size < 32 or size % 16:
        parser.error('Checkpoint has an invalid input size')
    model = NeedleDeepLabV3Plus()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    output = args.output_root / datetime.now().strftime('scene01_pointer_test_%Y%m%d_%H%M%S_%f') / 'overlays'
    output.mkdir(parents=True)
    with torch.inference_mode():
        for source in images:
            crop = load_image(argparse.Namespace(source=source))
            blend_labels(crop, predict_mask(model, crop, size))
            save_png(output / source.name, crop)
    print(f'Saved {len(images)} test-set overlays to {output.resolve()}')


if __name__ == '__main__':
    main()
