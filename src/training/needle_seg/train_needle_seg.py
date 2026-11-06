"""Fine-tune a binary needle segmenter from X-AnyLabeling polygon annotations."""
import argparse
import csv
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.paths import PROJECT_ROOT
from src.common.needle.dataset import NeedleDataset, load_splits
def train_epoch(model, encoder, loader, device, optimizer, scaler):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F

    model.train()
    # Retain the pretrained encoder's normalization statistics for this small dataset.
    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    total_loss = 0.0
    samples = 0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(images)
        logits = logits.float()
        probs = logits.sigmoid()
        dimensions = (1, 2, 3)
        overlap = (probs * masks).sum(dim=dimensions)
        dice = (2 * overlap + 1) / (probs.sum(dim=dimensions) + masks.sum(dim=dimensions) + 1)
        loss = F.binary_cross_entropy_with_logits(logits, masks) + (1 - dice.mean())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * len(images)
        samples += len(images)
    if dist.is_initialized():
        totals = torch.tensor([total_loss, samples], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        total_loss, samples = totals.tolist()
    return total_loss / samples


def evaluate(model, loader, device):
    import torch

    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for images, masks in loader:
            predicted = model(images.to(device, non_blocking=True)) >= 0
            target = masks.to(device, non_blocking=True) > 0.5
            tp += (predicted & target).sum().item()
            fp += (predicted & ~target).sum().item()
            fn += (~predicted & target).sum().item()
    return {
        'images': len(loader.dataset),
        'iou': tp / max(tp + fp + fn, 1),
        'dice': 2 * tp / max(2 * tp + fp + fn, 1),
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
    }


def run_training(args, parser, device, rank, world_size):
    import cv2
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from src.common.needle.needle_seg_model import NeedleDeepLabV3Plus

    distributed = world_size > 1
    main_rank = rank == 0
    batch_per_process = args.batch // world_size
    cv2.setNumThreads(0)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    annotations = args.dataset / 'xanylabel_json'
    try:
        splits = load_splits(args.dataset, args.split_images)
        datasets = {name: NeedleDataset(paths, annotations, args.imgsz, augment=name == 'train')
                    for name, paths in splits.items() if main_rank or name == 'train'}
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
        parser.error(str(exc))

    train_data = datasets['train']
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank) if distributed else None
    loader_options = {'batch_size': batch_per_process, 'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(
        train_data, sampler=sampler, shuffle=sampler is None, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        multiprocessing_context='spawn' if distributed and args.workers else None,
        **loader_options,
    )
    if main_rank:
        val_loader = DataLoader(datasets['val'], num_workers=0 if distributed else args.workers, **loader_options)
        test_loader = DataLoader(datasets['test'], num_workers=0 if distributed else args.workers, **loader_options)
        split_sizes = {name: len(paths) for name, paths in splits.items()}
        print(f'Processes: {world_size}, global batch: {args.batch}, batch per process: {batch_per_process}', flush=True)
        print('Split sizes:', split_sizes, flush=True)

    model = NeedleDeepLabV3Plus(args.encoder_weights).to(device)
    train_model = DistributedDataParallel(model, device_ids=[device.index]) if distributed else model
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 3e-5},
        {'params': [p for name, p in model.named_parameters() if not name.startswith('encoder.')], 'lr': 3e-4},
    ])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    log_file = nullcontext()
    if main_rank:
        run_dir = args.output_root / datetime.now().strftime('scene01_pointer_%Y%m%d_%H%M%S_%f')
        run_dir.mkdir(parents=True)
        best_path = run_dir / 'best.pt'
        log_file = (run_dir / 'results.csv').open('w', encoding='utf-8', newline='')
        print(f'Output: {run_dir}', flush=True)
    best_iou, best_epoch = -1.0, 0
    stop_flag = torch.zeros((), dtype=torch.int32, device=device) if distributed else None
    with log_file as stream:
        if main_rank:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'train_loss', 'val_iou', 'val_dice', 'val_precision', 'val_recall'])
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            loss = train_epoch(train_model, model.encoder, train_loader, device, optimizer, scaler)
            should_stop = False
            if main_rank:
                # The unwrapped model evaluates the full split without DDP collectives.
                val = evaluate(model, val_loader, device)
                writer.writerow([epoch, loss, val['iou'], val['dice'], val['precision'], val['recall']])
                stream.flush()
                print(f'Epoch {epoch}: loss={loss:.4f} val_iou={val["iou"]:.4f} val_dice={val["dice"]:.4f}', flush=True)
                if val['iou'] > best_iou:
                    best_iou, best_epoch = val['iou'], epoch
                    temporary = run_dir / 'best.pt.tmp'
                    torch.save({'state_dict': model.state_dict(), 'imgsz': args.imgsz,
                                'architecture': 'deeplabv3plus_resnet18',
                                'epoch': epoch, 'val_metrics': val}, temporary)
                    temporary.replace(best_path)
                should_stop = epoch - best_epoch >= args.patience
            if stop_flag is not None:
                stop_flag.fill_(int(should_stop))
                dist.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())
            if should_stop:
                break

    if not main_rank:
        return
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    report = {
        'best_epoch': best_epoch,
        'val': checkpoint['val_metrics'],
        'test': evaluate(model, test_loader, device),
        'split_sizes': split_sizes,
        'world_size': world_size,
        'global_batch': args.batch,
        'batch_per_process': batch_per_process,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Best checkpoint: {best_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT_ROOT / 'data/scene01_seg_min190')
    parser.add_argument('--split-images', type=Path, default=PROJECT_ROOT / 'data/scene01_yolo/images')
    parser.add_argument('--encoder-weights', type=Path, default=PROJECT_ROOT / 'weights/resnet18-f37072fd.pth')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--device', default='cuda:0', help='Device for single-process training')
    parser.add_argument('--local-rank', '--local_rank', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8, help='Global batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not args.encoder_weights.is_file():
        parser.error(f'Local torchvision ResNet-18 checkpoint not found: {args.encoder_weights}')
    if args.imgsz < 32 or args.imgsz % 16 or min(args.batch, args.epochs, args.patience) < 1 or args.workers < 0:
        parser.error('Use an image size divisible by 16 (at least 32), positive batch/epochs/patience and nonnegative workers')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 1 or args.batch % world_size:
        parser.error(f'Global --batch must be divisible by the number of processes ({world_size})')

    import torch
    import torch.distributed as dist

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; start the training container with a GPU')
    if world_size == 1:
        run_training(args, parser, device, rank=0, world_size=1)
        return

    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else args.local_rank
    if device.type != 'cuda' or local_rank is None or not 0 <= local_rank < torch.cuda.device_count():
        parser.error('Multi-GPU training requires torchrun and one visible CUDA device per local process')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(backend='nccl', device_id=device)
        run_training(args, parser, device, rank=dist.get_rank(), world_size=world_size)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
