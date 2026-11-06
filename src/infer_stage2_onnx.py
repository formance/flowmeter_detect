"""Run PaddleX DeepLabV3P ONNX on meter crops with ONNX Runtime CPU."""
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from infer_stage1 import DEFAULT_CONFIG, load_config, load_image, predict_dials


PROJECT = Path(__file__).resolve().parents[1]
INPUT_SHAPE = (1, 3, 512, 512)
CLASSES = {'pointer': 1, 'scale': 2}
PALETTE = np.array([[0, 0, 0], [255, 255, 0], [0, 255, 0]], dtype=np.uint8)  # BGR
OPACITY = np.array([0.0, 0.80, 0.45], dtype=np.float32)


def save_png(path, image):
    import cv2

    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Could not encode PNG: {path}')
    path.write_bytes(encoded.tobytes())


def prepare_input(crop, color_order):
    import cv2

    resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    if color_order == 'rgb':
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def decode_labels(outputs):
    # Prefer the exported class map when present. Otherwise decode scores.
    for output in outputs:
        if not np.issubdtype(output.dtype, np.integer):
            continue
        labels = output
        if labels.ndim == 4 and labels.shape[:2] == (1, 1):
            labels = labels[0, 0]
        elif labels.ndim == 4 and labels.shape[0] == labels.shape[-1] == 1:
            labels = labels[0, :, :, 0]
        elif labels.ndim == 3 and labels.shape[0] == 1:
            labels = labels[0]
        if labels.ndim == 2 and labels.size and labels.min() >= 0 and labels.max() <= 2:
            return labels.astype(np.uint8)
    for output in outputs:
        if output.ndim != 4 or output.shape[0] != 1 or not np.issubdtype(output.dtype, np.floating):
            continue
        if output.shape[1] == 3:
            return output[0].argmax(axis=0).astype(np.uint8)
        if output.shape[-1] == 3:
            return output[0].argmax(axis=-1).astype(np.uint8)
    shapes = ', '.join(f'{output.shape} {output.dtype}' for output in outputs)
    raise ValueError(f'Cannot decode a three-class segmentation output: {shapes}')


def segment_crop(session, input_name, crop, color_order, *, warmup=False):
    import cv2

    inputs = {input_name: prepare_input(crop, color_order)}
    if warmup:
        session.run(None, inputs)
    started = perf_counter()
    outputs = session.run(None, inputs)
    elapsed_ms = (perf_counter() - started) * 1000
    labels = decode_labels(outputs)
    height, width = crop.shape[:2]
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST), elapsed_ms


def blend_labels(image, labels):
    foreground = labels != 0
    classes = labels[foreground]
    opacity = OPACITY[classes, None]
    colors = PALETTE[classes]
    image[foreground] = np.rint(image[foreground] * (1 - opacity) + colors * opacity).astype(np.uint8)


def validate_crops(source_dir, output, session, input_name, color_order):
    import cv2

    images = sorted(path for path in source_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not images:
        raise ValueError(f'No PNG or JPEG crops found in {source_dir}')
    if len({image.stem.casefold() for image in images}) != len(images):
        raise ValueError('Crop filenames must have distinct stems')

    output.mkdir(parents=True)
    for folder in ('overlays', *CLASSES):
        (output / folder).mkdir()

    results, errors = [], []
    total_inference_ms = 0.0
    for source in images:
        try:
            crop = load_image(argparse.Namespace(source=source))
        except (OSError, ValueError, cv2.error) as exc:
            errors.append({'image': source.name, 'error': str(exc)})
            continue
        labels, elapsed_ms = segment_crop(session, input_name, crop, color_order, warmup=not results)
        total_inference_ms += elapsed_ms
        filename = source.with_suffix('.png').name
        pixels = {}
        for name, class_id in CLASSES.items():
            mask = labels == class_id
            pixels[name] = int(mask.sum())
            save_png(output / name / filename, mask.astype(np.uint8) * 255)
        blend_labels(crop, labels)
        save_png(output / 'overlays' / filename, crop)
        results.append({'image': source.name, 'foreground_pixels': pixels})

    mean_ms = total_inference_ms / len(results) if results else None
    inference = {
        'mean_ms': round(mean_ms, 3) if mean_ms is not None else None,
        'images_per_second': round(1000 / mean_ms, 3) if mean_ms else None,
    }
    report = {
        'total': len(images),
        'processed': len(results),
        'failed': len(errors),
        'onnx_inference': inference,
        'empty_masks': {
            name: [row['image'] for row in results if row['foreground_pixels'][name] == 0]
            for name in CLASSES
        },
        'results': results,
        'errors': errors,
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                       encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'processed': len(results), 'failed': len(errors),
                      'onnx_inference': inference}))
    return not errors


def detect_crops(args, image):
    import cv2
    import torch
    from ultralytics import YOLO

    config = load_config(args.dial_config)
    torch.set_num_threads(config['threads'])
    cv2.setNumThreads(config['threads'])
    model = YOLO(str(args.dial_weights.resolve()))
    if model.task != 'detect':
        raise ValueError('Stage 1 requires a YOLO Detect model')
    return predict_dials(model, image, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--source', type=Path, help='Local image; defaults to data/test.jpg')
    sources.add_argument('--source-dir', type=Path, help='Directory of pre-cropped meter images')
    sources.add_argument('--source-base64', help='Base64 image or image data URI')
    sources.add_argument('--base64-file', help='Base64 text file or - for stdin')
    parser.add_argument('--crop', action='store_true', help='Input is already cropped to one meter')
    parser.add_argument('--onnx', type=Path, default=PROJECT / 'weights/meter_seg_model.onnx')
    parser.add_argument('--dial-weights', type=Path, default=PROJECT / 'weights/gauge_detection_model.pt')
    parser.add_argument('--dial-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--color-order', choices=('bgr', 'rgb'), default='bgr')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='New directory for output images')
    args = parser.parse_args()
    if all(value is None for value in (args.source, args.source_dir, args.source_base64, args.base64_file)):
        args.source = PROJECT / 'data/test.jpg'
    if not args.onnx.is_file():
        parser.error(f'ONNX model not found: {args.onnx}')
    if args.source_dir is not None and not args.source_dir.is_dir():
        parser.error(f'Crop directory not found: {args.source_dir}')
    if args.source_dir is None and not args.crop and not args.dial_weights.is_file():
        parser.error(f'Stage 1 weights not found: {args.dial_weights}; use --crop for a pre-cropped image')
    if args.threads < 1:
        parser.error('--threads must be positive')

    import cv2
    try:
        import onnxruntime as ort
    except ImportError:
        parser.error('Install onnxruntime in the Python environment running this script')

    output = args.output or PROJECT / 'outputs' / datetime.now().strftime('stage2_onnx_%Y%m%d_%H%M%S_%f')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        session = ort.InferenceSession(str(args.onnx.resolve()), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
            raise ValueError('Expected one float32 NCHW image input in the ONNX model')
        if any(isinstance(actual, int) and actual > 0 and actual != expected
               for actual, expected in zip(inputs[0].shape, INPUT_SHAPE, strict=True)):
            raise ValueError(f'ONNX input shape {inputs[0].shape} is incompatible with {INPUT_SHAPE}')
        if args.source_dir is not None:
            if not validate_crops(args.source_dir, output, session, inputs[0].name, args.color_order):
                raise SystemExit(1)
            return
        image = load_image(args)
        height, width = image.shape[:2]
        if args.crop:
            detections, annotated = [{'bbox': [0, 0, width, height]}], image.copy()
        else:
            detections, annotated = detect_crops(args, image)
        output.mkdir(parents=True)
        dials = []
        for index, detection in enumerate(detections, start=1):
            left, top, right, bottom = detection['bbox']
            x1, y1 = max(0, math.floor(left)), max(0, math.floor(top))
            x2, y2 = min(width, math.ceil(right)), min(height, math.ceil(bottom))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            labels, _ = segment_crop(session, inputs[0].name, crop, args.color_order)
            stem = f'dial_{index:02d}'
            save_png(output / f'{stem}_crop.png', crop)
            mask_files = {}
            for name, class_id in CLASSES.items():
                filename = f'{stem}_{name}.png'
                save_png(output / filename, (labels == class_id).astype(np.uint8) * 255)
                mask_files[name] = filename
            preview = crop.copy()
            blend_labels(preview, labels)
            save_png(output / f'{stem}_segmented.png', preview)
            blend_labels(annotated[y1:y2, x1:x2], labels)
            dials.append({'crop_bbox': [x1, y1, x2, y2], 'masks': mask_files})
        save_png(output / 'stage2.png', annotated)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    print(json.dumps({'output': str(output.resolve()), 'dials': dials}, indent=2))


if __name__ == '__main__':
    main()
