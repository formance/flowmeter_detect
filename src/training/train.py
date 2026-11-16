"""Fine-tune one DeepLabV3+ to output background, pointer and scale ticks."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler

from src.common.dataset import SegmentationDataset
from src.common.metrics import Metrics,segmentation_loss
from src.common.model import DeepLabV3Plus

PROJECT=Path(__file__).resolve().parents[2]


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();metrics=Metrics(device)
    for images,labels in loader:
        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
        metrics.update(model(images),labels)
    return metrics.compute()


def train_epoch(model,encoder,loader,device,optimizer,scaler,weights,amp,dice_weight):
    model.train()
    # Small dataset: keep the pretrained encoder normalization statistics.
    for module in encoder.modules():
        if isinstance(module,torch.nn.BatchNorm2d):module.eval()
    total=torch.zeros(2,dtype=torch.float64,device=device)
    for images,labels in loader:
        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=amp):logits=model(images)
        loss=segmentation_loss(logits,labels,weights,dice_weight)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update()
        total[0]+=loss.detach().double()*len(images);total[1]+=len(images)
    if dist.is_initialized():dist.all_reduce(total)
    return float((total[0]/total[1]).item())


def save_checkpoint(path,model,args,epoch,metrics,dataset_hash):
    temporary=path.with_suffix('.pt.tmp')
    torch.save({'architecture':'deeplabv3plus_pointer_scale','classes':['background','pointer','scale_tick'],
                'state_dict':model.state_dict(),'model_config':model.model_config,'imgsz':args.imgsz,
                'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
                'input_resize':'letterbox','ignore_index':255,'tick_line_width':1,
                'epoch':epoch,'val_metrics':metrics,'dataset_manifest_sha256':dataset_hash,
                'config':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}},temporary)
    temporary.replace(path)


def run(args,device,rank,world_size):
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    torch.set_num_threads(args.cpu_threads)
    train_data=SegmentationDataset(args.dataset,'train',args.imgsz,augment=True)
    sampler=DistributedSampler(train_data,num_replicas=world_size,rank=rank,seed=args.seed) if world_size>1 else None
    train_loader=DataLoader(train_data,batch_size=args.batch//world_size,shuffle=sampler is None,sampler=sampler,
                            num_workers=args.workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,
                            persistent_workers=args.workers>0)
    val_loader=None
    if rank==0:
        val_data=SegmentationDataset(args.dataset,'val',args.imgsz)
        val_loader=DataLoader(val_data,batch_size=args.batch//world_size,num_workers=args.workers,
                              pin_memory=device.type=='cuda',worker_init_fn=seed_worker)
    model=DeepLabV3Plus(args.backbone,args.output_stride,args.decoder_channels,args.encoder_weights).to(device)
    training=DDP(model,device_ids=[device.index] if device.type=='cuda' else None) if world_size>1 else model
    optimizer=torch.optim.AdamW([
        {'params':model.encoder.parameters(),'lr':args.encoder_lr},
        {'params':[p for name,p in model.named_parameters() if not name.startswith('encoder.')],'lr':args.lr}],
        weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:(1-min(epoch,args.epochs)/args.epochs)**.9)
    amp=device.type=='cuda' and not args.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    weights=torch.tensor(args.class_weights,dtype=torch.float32,device=device)
    run_dir=None;writer=None;stream=None
    dataset_hash=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    if rank==0:
        run_dir=args.output_root/datetime.now(timezone(timedelta(hours=8))).strftime('pointer_scale_%Y%m%d_%H%M%S_%f')
        run_dir.mkdir(parents=True)
        config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
        (run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
        dataset_manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
        (run_dir/'dataset_summary.json').write_text(json.dumps({key:dataset_manifest[key] for key in ('counts','scenes','seed','policy')},ensure_ascii=False,indent=2),encoding='utf-8')
        stream=(run_dir/'results.csv').open('w',newline='',encoding='utf-8');writer=csv.writer(stream)
        writer.writerow(['epoch','train_loss','val_foreground_miou','pointer_iou','scale_iou','pointer_dice','scale_dice','scale_f1_tolerance_1px'])
        print(f'Output: {run_dir}\nTrain: {len(train_data)}  Val: {len(val_loader.dataset)}  GPUs/processes: {world_size}',flush=True)
    best=-1.;best_epoch=0;stop=torch.zeros((),dtype=torch.int32,device=device)
    try:
        for epoch in range(1,args.epochs+1):
            if sampler:sampler.set_epoch(epoch)
            loss=train_epoch(training,model.encoder,train_loader,device,optimizer,scaler,weights,amp,args.dice_weight)
            scheduler.step()
            if rank==0:
                metrics=evaluate(model,val_loader,device)
                pointer,scale=metrics['pointer'],metrics['scale_tick'];score=metrics['foreground_miou']
                writer.writerow([epoch,loss,score,pointer['iou'],scale['iou'],pointer['dice'],scale['dice'],scale['f1_tolerance_1px']]);stream.flush()
                save_checkpoint(run_dir/'last.pt',model,args,epoch,metrics,dataset_hash)
                if score>best:
                    best,best_epoch=score,epoch
                    save_checkpoint(run_dir/'best.pt',model,args,epoch,metrics,dataset_hash)
                    (run_dir/'best_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
                print(f'Epoch {epoch}/{args.epochs}: loss={loss:.4f} foreground_mIoU={score:.4f} pointer_IoU={pointer["iou"]} scale_IoU={scale["iou"]}',flush=True)
                stop.fill_(int(epoch-best_epoch>=args.patience))
            if world_size>1:dist.broadcast(stop,0)
            if bool(stop.item()):break
        if rank==0:print(f'Best epoch: {best_epoch}; foreground mIoU: {best:.4f}; checkpoint: {run_dir/"best.pt"}',flush=True)
    finally:
        if stream:stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=PROJECT/'data/pointer_scale_seg')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'outputs')
    parser.add_argument('--backbone',choices=['resnet18','resnet34'],default='resnet18')
    parser.add_argument('--encoder-weights',default=str(PROJECT/'weights/resnet18-f37072fd.pth'),help='Local torchvision encoder state_dict; imagenet explicitly downloads, none is for smoke tests')
    parser.add_argument('--output-stride',type=int,choices=[8,16],default=8)
    parser.add_argument('--decoder-channels',type=int,default=128)
    parser.add_argument('--imgsz',type=int,default=768)
    parser.add_argument('--batch',type=int,default=4,help='Global batch size, divided across all processes')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=20)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--encoder-lr',type=float,default=3e-5)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--class-weights',type=float,nargs=3,default=[.2,1.,2.])
    parser.add_argument('--dice-weight',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=20261113)
    parser.add_argument('--no-amp',action='store_true')
    args=parser.parse_args()
    world_size=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if args.imgsz<64 or args.imgsz%16 or min(args.batch,args.epochs,args.patience,args.cpu_threads,args.decoder_channels)<1 or args.workers<0 or args.batch%world_size:
        parser.error('Use imgsz divisible by 16 (>=64), positive parameters, and global batch divisible by world size')
    if (any(not math.isfinite(v) or v<=0 for v in [args.lr,args.encoder_lr,*args.class_weights])
            or any(not math.isfinite(v) or v<0 for v in (args.dice_weight,args.weight_decay))):
        parser.error('Learning rates and class weights must be finite and positive')
    if args.encoder_weights not in ('none','imagenet') and not Path(args.encoder_weights).is_file():
        parser.error(f'Local encoder weights not found: {args.encoder_weights}; see docs/TRAINING.md')
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():parser.error('CUDA unavailable; mount a GPU or use --device cpu')
        if world_size>1:device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
    try:
        if world_size>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
        run(args,device,rank,world_size)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
