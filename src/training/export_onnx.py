"""Export trained pointer/scale DeepLabV3+ logits to ONNX."""
import argparse
import json
from pathlib import Path

import torch
from src.common.model import load_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--opset',type=int,default=17)
    args=parser.parse_args()
    import onnx
    model,checkpoint=load_checkpoint(args.checkpoint)
    model.eval();size=checkpoint['imgsz']
    output=args.output or args.checkpoint.with_suffix('.onnx')
    output.parent.mkdir(parents=True,exist_ok=True)
    tensor=torch.zeros(1,3,size,size)
    with torch.inference_mode():
        torch.onnx.export(model,tensor,str(output),input_names=['images'],output_names=['logits'],opset_version=args.opset,dynamo=False)
    exported=onnx.load(str(output));onnx.checker.check_model(exported)
    output.with_suffix('.json').write_text(json.dumps({'classes':checkpoint['classes'],'input_shape':[1,3,size,size],
        'output_shape':[1,3,size,size],'resize':'letterbox','normalization':checkpoint['normalization'],
        'postprocess':'argmax logits along class dimension; remove letterbox padding then resize labels with nearest interpolation',
        'tick_line_width':checkpoint.get('tick_line_width',1)},indent=2)+'\n',encoding='utf-8')
    print(f'ONNX exported and checked: {output}')


if __name__=='__main__':main()
