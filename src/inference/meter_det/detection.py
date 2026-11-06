"""Detector configuration and Ultralytics prediction for meter faces."""
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name('stage1.json')


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
