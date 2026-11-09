"""Read a calibrated flowmeter using three ONNX models on CPU."""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from src.common.image_input import decode_base64_image
from src.common.image_preprocessing import enhance_meter_crop
from src.common.paths import PROJECT_ROOT
from src.common.visualization import blend_labels, save_png
from src.inference.flowmeter.flowmeter_reading import load_calibrations, read_dial
from src.inference.meter_det.detection import DEFAULT_CONFIG, load_config
from src.common.needle.normalization import RGB_MEAN, RGB_STD
from src.inference.scale_seg.model import segment_crop


def find_model(directory, *names):
    for name in names:
        for parent in (directory, directory / 'onnx'):
            candidate = parent / name
            if candidate.is_file():
                return candidate
    raise ValueError(f'No model found under {directory}: {", ".join(names)}')


def model_input(session, fallback):
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or len(inputs[0].shape) != 4:
        raise ValueError('Expected one float32 NCHW image input per model')
    name = inputs[0].name
    batch, channels, height, width = inputs[0].shape
    if (type(batch) is int and batch != 1) or (type(channels) is int and channels != 3):
        raise ValueError(f'Model input {name} must support [1, 3, H, W]')
    height = height if type(height) is int else fallback
    width = width if type(width) is int else fallback
    if min(height, width) < 1:
        raise ValueError(f'Model input {name} has invalid image dimensions')
    return name, height, width


def nms(boxes, scores, iou):
    order = np.argsort(-scores, kind='stable')
    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    keep = []
    while order.size and len(keep) < 300:
        index, remaining = order[0], order[1:]
        keep.append(index)
        left = np.maximum(boxes[index, :2], boxes[remaining, :2])
        right = np.minimum(boxes[index, 2:], boxes[remaining, 2:])
        intersection = np.prod(np.maximum(right - left, 0), axis=1)
        overlap = intersection / (areas[index] + areas[remaining] - intersection)
        order = remaining[overlap <= iou]
    return keep


def detect_dials(session, model_input, image, config):
    import cv2

    name, height, width = model_input
    source_h, source_w = image.shape[:2]
    gain = min(width / source_w, height / source_h)
    new_w, new_h = max(1, round(source_w * gain)), max(1, round(source_h * gain))
    left, top = (width - new_w) // 2, (height - new_h) // 2
    padded = cv2.copyMakeBorder(cv2.resize(image, (new_w, new_h)), top, height - new_h - top,
                                left, width - new_w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    outputs = session.run(None, {name: tensor})
    if len(outputs) != 1 or outputs[0].ndim != 3 or outputs[0].shape[0] != 1:
        raise ValueError('Detector must be a single-class YOLO ONNX export with NMS disabled')
    rows = outputs[0][0]
    if rows.shape[0] == 5:
        rows = rows.T
    if rows.shape[1] != 5:
        raise ValueError(f'Expected raw detector output [1,5,N] or [1,N,5], got {outputs[0].shape}')
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= config['conf']) & (rows[:, 2:4] > 0).all(axis=1)]
    half_size = rows[:, 2:4] / 2
    boxes = np.concatenate((rows[:, :2] - half_size, rows[:, :2] + half_size), axis=1)
    detections = []
    for index in nms(boxes, rows[:, 4], config['iou']):
        x1, y1, x2, y2 = (boxes[index] - [left, top, left, top]) / gain
        bbox = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                min(source_w, math.ceil(x2)), min(source_h, math.ceil(y2))]
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            detections.append({'bbox': bbox, 'confidence': float(rows[index, 4])})
    return detections


def segment_needle(session, model_input, crop):
    import cv2

    name, height, width = model_input
    rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = (rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    outputs = session.run(None, {name: tensor})
    if len(outputs) != 1 or outputs[0].shape != (1, 1, height, width) or not np.isfinite(outputs[0]).all():
        raise ValueError('Expected one finite needle_logits tensor shaped [1,1,H,W]')
    mask = (outputs[0][0, 0] >= 0).astype(np.uint8)
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def resolve_input_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def sources_from_args(args):
    if args.source_dir is not None:
        source_dir = resolve_input_path(args.source_dir)
        if not source_dir.is_dir():
            raise ValueError(f'Image directory not found: {source_dir}. '
                             f'Project root: {PROJECT_ROOT}. Check the dataset path and container mount.')
        paths = sorted(path for path in source_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'})
        return [(str(path), path) for path in paths]
    if args.source:
        return [(str(path), resolve_input_path(path)) for path in args.source]
    if args.source_base64:
        return [(f'base64_{i}', value) for i, value in enumerate(args.source_base64, start=1)]

    sources = []
    for filename in args.base64_file:
        content = (sys.stdin.read() if filename == '-'
                   else resolve_input_path(filename).read_text(encoding='utf-8-sig'))
        values = json.loads(content) if content.lstrip().startswith('[') else [content]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError('Base64 JSON input must be an array of image strings')
        sources.extend((f'{filename}#{i}', value) for i, value in enumerate(values, start=1))
    return sources


def decode_image(source):
    import cv2

    raw = source.read_bytes() if isinstance(source, Path) else decode_base64_image(source)
    if not raw:
        raise ValueError('Empty image file')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, nargs='+', help='One or more image files')
    inputs.add_argument('--source-dir', type=Path, help='Directory of images')
    inputs.add_argument('--source-base64', nargs='+', help='Base64 strings or image data URIs')
    inputs.add_argument('--base64-file', nargs='+', help='Text or JSON array of Base64 images; use - for stdin')
    parser.add_argument('--weights-dir', type=Path, default=PROJECT_ROOT / 'weights')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--calibration', type=Path, default=Path(__file__).with_name('reading_080.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.base64_file and args.base64_file.count('-') > 1:
        parser.error('Standard input can be used once')
    try:
        config = load_config(args.config)
        sources = sources_from_args(args)
        if not sources:
            raise ValueError('No input images')
        paths = (
            find_model(args.weights_dir, 'gauge_det_best.onnx', 'det_best.onnx'),
            find_model(args.weights_dir, 'meter_seg_model.onnx'),
            find_model(args.weights_dir, 'needle_seg_best.onnx'),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import cv2
    import onnxruntime as ort

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = config['threads']
    det, scale, needle = (ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                          for path in paths)
    det_input = model_input(det, config['imgsz'])
    scale_input = model_input(scale, 512)
    needle_input = model_input(needle, 512)
    if scale_input[1:] != (512, 512):
        parser.error('meter_seg_model.onnx must support 512×512 inputs')

    def scale_mask(crop):
        labels, _ = segment_crop(scale, scale_input[0], crop, 'bgr')
        return labels == 2

    try:
        calibrations = load_calibrations(args.calibration, scale_mask)
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as exc:
        parser.error(str(exc))
    output = args.output or PROJECT_ROOT / 'outputs' / datetime.now().strftime('flowmeter_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    results = []
    for index, (name, source) in enumerate(sources, start=1):
        try:
            image = decode_image(source)
        except (OSError, ValueError, cv2.error) as exc:
            results.append({'source': name, 'status': 'input_error', 'inference_ms': None, 'error': str(exc)})
            continue
        started = perf_counter()
        detections = detect_dials(det, det_input, image, config)
        inference_seconds = perf_counter() - started
        annotated = image.copy()
        folder = output / f'image_{index:04d}'
        folder.mkdir()
        for number, dial in enumerate(detections, start=1):
            started = perf_counter()
            x1, y1, x2, y2 = dial['bbox']
            crop = image[y1:y2, x1:x2]
            pointer = segment_needle(needle, needle_input, crop)
            ticks = scale_mask(crop)
            reading = read_dial(calibrations, crop, pointer, ticks)
            preprocessing = 'original'
            if reading['reading'] is None and pointer.any():
                enhanced_crop = enhance_meter_crop(crop)
                enhanced_ticks = scale_mask(enhanced_crop)
                enhanced_reading = read_dial(calibrations, enhanced_crop, pointer, enhanced_ticks, enhanced=True)
                if enhanced_reading['reading'] is not None:
                    reading, ticks = enhanced_reading, enhanced_ticks
                    preprocessing = 'denoise_clahe'
            dial.update(reading)
            dial['scale_preprocessing'] = preprocessing
            inference_seconds += perf_counter() - started
            for label, mask in (('pointer', pointer), ('scale', ticks)):
                path = folder / f'dial_{number:02d}_{label}.png'
                save_png(path, mask.astype(np.uint8) * 255)
                dial[f'{label}_mask'] = path.relative_to(output).as_posix()
            labels = ticks.astype(np.uint8) * 2
            labels[pointer] = 1
            blend_labels(annotated[y1:y2, x1:x2], labels)
        for dial in detections:
            x1, y1, x2, y2 = dial['bbox']
            color = (255, 170, 0)
            cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            caption = f'{dial["reading"]:.1f}' if dial['reading'] is not None else 'Reading unavailable'
            cv2.putText(annotated, caption, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)
        preview = folder / 'overlay.png'
        save_png(preview, annotated)
        status = 'no_dial'
        if detections:
            status = 'ok' if any(dial['reading'] is not None for dial in detections) else 'reading_unavailable'
        results.append({'source': name, 'status': status, 'inference_ms': round(inference_seconds * 1000, 3),
                        'overlay': preview.relative_to(output).as_posix(), 'dials': detections})
    report = {'calibration': str(args.calibration), 'models': [str(path) for path in paths], 'images': results}
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Results: {output.resolve() / "results.json"}')
    if any(result['status'] == 'input_error' for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
