"""Fine-tune the single-class gauge detector on the Scene01 YOLO dataset."""
import argparse
import os
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

from src.common.paths import PROJECT_ROOT


def reserve_run_dir(output_root):
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    base_name = f'scene01_detect_{timestamp}'
    output_root.mkdir(parents=True, exist_ok=True)
    for index in count():
        run_dir = output_root / (f'{base_name}_{index}' if index else base_name)
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT_ROOT / 'data/scene01_yolo/data.yaml')
    parser.add_argument('--weights', type=Path, default=PROJECT_ROOT / 'weights/gauge_detection_model.pt')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--device', default='0', help='CUDA index inside the training container')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr0', type=float, default=0.001)
    args = parser.parse_args()

    if args.data.suffix.lower() not in {'.yaml', '.yml'} or not args.data.is_file():
        parser.error('--data must point to an existing dataset YAML file')
    if args.weights.suffix.lower() != '.pt' or not args.weights.is_file():
        parser.error('--weights must point to an existing PyTorch .pt checkpoint')
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error('--epochs and --batch must be positive, and --imgsz must be at least 32')
    if args.workers < 0 or args.patience < 0 or not 0 < args.lr0 <= 1:
        parser.error('--workers and --patience must be nonnegative, and --lr0 must be in (0, 1]')

    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    if model.task != 'detect' or len(model.names) != 1:
        parser.error(f'Expected a single-class Detect checkpoint, got task={model.task}, names={model.names}')

    run_dir = reserve_run_dir(args.output_root.resolve())
    print(f'Training output: {run_dir}', flush=True)
    model.train(
        data=str(args.data.resolve()),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        optimizer='AdamW',
        lr0=args.lr0,
        seed=0,
        resume=False,
        amp=False,  # Ultralytics 8.3.162 otherwise tries to load YOLO11n for its AMP check.
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
    )
    print(f'Best checkpoint: {model.trainer.best}')


if __name__ == '__main__':
    main()
