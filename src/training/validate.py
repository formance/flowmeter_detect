"""Evaluate a trained three-class model without mixing test data into training."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.common.dataset import SegmentationDataset
from src.common.model import load_checkpoint
from src.training.train import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('data/pointer_scale_seg'))
    parser.add_argument('--split',choices=['val','test'],default='test')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();device=torch.device(args.device)
    model,checkpoint=load_checkpoint(args.checkpoint,device)
    dataset=SegmentationDataset(args.dataset,args.split,checkpoint['imgsz'])
    metrics=evaluate(model,DataLoader(dataset,batch_size=args.batch,num_workers=args.workers),device)
    report={'split':args.split,'images':len(dataset),'checkpoint':str(args.checkpoint),'epoch':checkpoint['epoch'],'metrics':metrics}
    output=args.output or args.checkpoint.parent/(args.split+'_metrics.json')
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
