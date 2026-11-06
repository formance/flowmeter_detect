"""Export the meter detector and trained needle segmenter to CPU-compatible ONNX."""
import argparse
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPSET = 17


def export_detector(weights, staging, image_size):
    from ultralytics import YOLO

    # Ultralytics writes the ONNX beside its input checkpoint.
    staged_weights = staging / 'gauge_det_best.pt'
    shutil.copyfile(weights, staged_weights)
    model = YOLO(str(staged_weights))
    if model.task != 'detect' or len(model.names) != 1:
        raise ValueError('Expected a single-class YOLO Detect checkpoint for --det-weights')
    size = image_size if image_size is not None else model.overrides.get('imgsz', 640)
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError('Detector image size must be a multiple of 32; set --det-imgsz explicitly')
    return Path(model.export(
        format='onnx', imgsz=size, device='cpu', batch=1, opset=OPSET,
        dynamic=False, simplify=False, half=False, nms=False,
    ))


def export_needle(weights, staging):
    import torch
    from needle_seg_model import NeedleDeepLabV3Plus

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('architecture') != 'deeplabv3plus_resnet18':
        raise ValueError('Expected a DeepLabV3+ checkpoint from train_needle_seg.py for --needle-weights')
    size = checkpoint.get('imgsz')
    if type(size) is not int or size < 32 or size % 16:
        raise ValueError('Needle checkpoint has an invalid input size')
    model = NeedleDeepLabV3Plus()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    example = torch.zeros((1, 3, size, size), dtype=torch.float32)
    destination = staging / 'needle_seg_best.onnx'
    torch.onnx.export(
        model, (example,), str(destination), opset_version=OPSET, dynamo=False,
        input_names=['images'], output_names=['needle_logits'],
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--det-weights', type=Path, default=PROJECT_ROOT / 'weights/gauge_det_best.pt')
    parser.add_argument('--needle-weights', type=Path, default=PROJECT_ROOT / 'weights/needle_seg_best.pt')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'weights/onnx')
    parser.add_argument('--det-imgsz', type=int, help='Override the detector checkpoint input size')
    args = parser.parse_args()
    for weights in (args.det_weights, args.needle_weights):
        if not weights.is_file() or weights.suffix.lower() != '.pt':
            parser.error(f'Expected an existing .pt checkpoint: {weights}')

    output = args.output_dir.resolve()
    destinations = (output / 'gauge_det_best.onnx', output / 'needle_seg_best.onnx')
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            parser.error(f'Output already exists: {destination}; choose a new --output-dir')

    try:
        import onnx
    except ImportError:
        parser.error('Install onnx==1.17.0 or run scripts/export_models_onnx.sh --install-onnx')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.onnx-export-', dir=output) as temporary:
        staging = Path(temporary)
        models = (export_detector(args.det_weights, staging, args.det_imgsz),
                  export_needle(args.needle_weights, staging))
        for model in models:
            onnx.checker.check_model(str(model))

        published = []
        try:
            for source, destination in zip(models, destinations, strict=True):
                with source.open('rb') as exported, destination.open('xb') as target:
                    published.append(destination)
                    shutil.copyfileobj(exported, target)
        except (OSError, KeyboardInterrupt):
            for destination in published:
                destination.unlink()
            raise

    for destination in destinations:
        print(f'Exported and checked: {destination}')


if __name__ == '__main__':
    main()
