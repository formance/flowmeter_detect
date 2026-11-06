"""Preprocess crops and decode the three-class ONNX scale segmenter."""
from time import perf_counter

import numpy as np


INPUT_SHAPE = (1, 3, 512, 512)
CLASSES = {'pointer': 1, 'scale': 2}


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
