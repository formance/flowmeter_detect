"""Detect meter faces in one image and save boxes in the source image coordinates."""
import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path

from src.common.image_input import load_image
from src.inference.meter_det.detection import DEFAULT_CONFIG, load_config, predict_dials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='Local image file')
    inputs.add_argument('--source-base64', help='Base64 image or image data URI')
    inputs.add_argument('--base64-file', help='Base64 text file or - to read stdin')
    parser.add_argument('--weights', type=Path, required=True, help='Local YOLO Detect .pt or .onnx model')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=Path('outputs/stage1.jpg'), help='Annotated image path')
    args = parser.parse_args()

    if not args.weights.is_file() or args.weights.suffix.lower() not in ('.pt', '.onnx'):
        parser.error('--weights must point to an existing .pt or .onnx detection model')
    suffix = args.output.suffix.lower()
    if suffix not in ('.jpg', '.jpeg', '.png'):
        parser.error('--output must end in .jpg, .jpeg or .png')
    if args.source is not None and args.output.resolve() == args.source.resolve():
        parser.error('--output must use a different path from --source')

    import cv2

    try:
        config = load_config(args.config)
        image = load_image(args)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))

    if args.weights.suffix.lower() == '.onnx':
        missing = [name for name in ('onnx', 'onnxruntime') if find_spec(name) is None]
        if missing:
            parser.error(f'ONNX inference requires installed packages: {", ".join(missing)}')

    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(config['threads'])
    cv2.setNumThreads(config['threads'])
    model = YOLO(str(args.weights.resolve()), task='detect')
    if model.task != 'detect':
        parser.error(f'Expected a detection model, got {model.task}')

    detections, annotated = predict_dials(model, image, config)
    try:
        ok, encoded = cv2.imencode(suffix, annotated)
        if not ok:
            raise OSError('Cannot encode annotated image')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as exc:
        parser.error(f'Cannot save annotated image: {exc}')
    print(json.dumps({
        'annotated_image': str(args.output.resolve()),
        'detections': detections,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
