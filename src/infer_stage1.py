"""Detect meter faces in one image and save boxes in the source image coordinates."""
import argparse
import json
import os
import sys
from importlib.util import find_spec
from pathlib import Path

from image_input import decode_base64_image


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'stage1.json'


def load_config(path):
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a JSON object')
    for key in ('conf', 'iou'):
        value = config.get(key)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be a number between 0 and 1')
    for key, default, minimum in (('imgsz', 640, 32), ('threads', 4, 1)):
        value = config.setdefault(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    return config


def load_image(args):
    import cv2
    import numpy as np

    if args.source is not None:
        raw = args.source.read_bytes()
    else:
        encoded = args.source_base64
        if args.base64_file is not None:
            encoded = (sys.stdin.read() if args.base64_file == '-'
                       else Path(args.base64_file).read_text(encoding='utf-8-sig'))
        raw = decode_base64_image(encoded)
    if not raw:
        raise ValueError('Empty image')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode source image')
    return image


def predict_dials(model, image, config):
    result = model.predict(
        source=image, device='cpu', imgsz=config['imgsz'], half=False,
        conf=config['conf'], iou=config['iou'], verbose=False, save=False,
    )[0]
    detections = [
        {'label': result.names[int(box.cls.item())],
         'confidence': float(box.conf.item()),
         'bbox': box.xyxy[0].cpu().tolist()}
        for box in result.boxes
    ]
    return detections, result.plot()


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
