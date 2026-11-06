"""Evaluate a single-class gauge detector at fixed confidence and IoU thresholds."""
import argparse
import json
import math
from pathlib import Path


def load_test_split(dataset):
    annotation_path = dataset / 'annotations' / 'instance_test.json'
    data = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    categories = data['categories']
    if len(categories) != 1 or categories[0]['name'].lower() != 'meter':
        raise ValueError('Expected one COCO category named meter')
    category_id = categories[0]['id']
    images = {im['id']: im for im in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('Duplicate image ids')
    if not images:
        raise ValueError('Empty test split')
    labels = {key: [] for key in images}
    for ann in data['annotations']:
        if ann['image_id'] not in images or ann['category_id'] != category_id:
            raise ValueError(f'Unknown image/category in annotation {ann.get("id")}')
        if ann.get('iscrowd', 0):
            raise ValueError('Crowd annotations require COCO evaluation; not supported here')
        im = images[ann['image_id']]
        width, height = float(im['width']), float(im['height'])
        x, y, w, h = map(float, ann['bbox'])
        if not all(math.isfinite(v) for v in (width, height, x, y, w, h)) or min(width, height, w, h) <= 0:
            raise ValueError(f'Invalid dimensions in annotation {ann.get("id")}')
        x1, y1 = max(0., x), max(0., y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Box outside image in annotation {ann.get("id")}')
        labels[ann['image_id']].append((x1, y1, x2, y2))
    test_dir = (dataset / 'test').resolve()
    samples = []
    for image_id, im in images.items():
        source = (test_dir / im['file_name']).resolve()
        if not source.is_relative_to(test_dir) or not source.is_file():
            raise ValueError(f'Missing or invalid test image: {im["file_name"]}')
        samples.append((image_id, source, (im['height'], im['width']), labels[image_id]))
    return samples


def box_iou(a, b):
    overlap_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_width * overlap_height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_ious(predictions, targets, threshold):
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            iou = box_iou(prediction, target)
            if iou > 0 and iou >= threshold:
                candidates.append((iou, pred_index, target_index))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    used_predictions, used_targets = set(), set()
    matched = []
    for iou, pred_index, target_index in candidates:
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
        matched.append(iou)
    return matched


def evaluate(model, samples, output_dir, args):
    import cv2

    target_count = prediction_count = true_positives = 0
    iou_sum = 0.0
    output_dir.mkdir(parents=True)
    for start in range(0, len(samples), args.batch):
        batch = samples[start:start + args.batch]
        results = model.predict(
            source=[str(source) for _, source, _, _ in batch],
            device=args.device, imgsz=args.imgsz, conf=args.conf, iou=0.7,
            half=False, stream=True, verbose=False, save=False,
        )
        for (image_id, source, shape, targets), result in zip(batch, results, strict=True):
            if Path(result.path).resolve() != source or tuple(result.orig_shape) != shape:
                raise ValueError(f'Prediction image or dimensions do not match annotations: {source}')
            predictions = result.boxes.xyxy.cpu().tolist()
            matched = match_ious(predictions, targets, args.iou)
            target_count += len(targets)
            prediction_count += len(predictions)
            true_positives += len(matched)
            iou_sum += sum(matched)
            destination = output_dir / f'image_{image_id}.jpg'
            if not cv2.imwrite(str(destination), result.plot()):
                raise OSError(f'Could not write image: {destination}')
    return {
        'recall': true_positives / target_count if target_count else 0.0,
        'mIoU': iou_sum / true_positives if true_positives else 0.0,
        'precision': true_positives / prediction_count if prediction_count else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/meter_det_validation'))
    p.add_argument('--device', default='0', help='0 for GPU, cpu for CPU')
    p.add_argument('--conf', type=float, default=0.5,
                   help='Minimum prediction confidence (default: 0.5)')
    p.add_argument('--iou', type=float, default=0.5,
                   help='Minimum prediction-to-annotation IoU for a match (default: 0.5)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        p.error('--conf and --iou must be between 0 and 1')
    if not args.weights.is_file():
        p.error('Provide an existing local --weights .pt file')
    if args.imgsz < 32 or args.batch < 1 or args.threads < 1:
        p.error('Invalid imgsz, batch or threads')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Output directory is not empty; choose a new --output path')
    try:
        samples = load_test_split(args.dataset.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel('ERROR')
    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights.resolve()), task='detect')
    if model.task != 'detect' or len(model.names) != 1:
        p.error(f'Expected a single-class detection model; got task={model.task}, names={model.names}')
    print(json.dumps(evaluate(model, samples, output / 'predictions', args), indent=2))


if __name__ == '__main__':
    main()
