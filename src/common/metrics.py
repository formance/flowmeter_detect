"""Class-balanced losses and foreground metrics for very thin scale lines."""
import torch
from torch.nn import functional as F

NAMES=('background','pointer','scale_tick')


def segmentation_loss(logits,target,class_weights,dice_weight=1.):
    logits=logits.float();valid=target!=255
    ce=F.cross_entropy(logits,target,weight=class_weights,ignore_index=255)
    one_hot=F.one_hot(target.clamp(0,2),3).permute(0,3,1,2).float()*valid[:,None]
    probabilities=logits.softmax(1)*valid[:,None]
    dimensions=(0,2,3)
    intersection=(probabilities*one_hot).sum(dimensions)
    dice=(2*intersection+1)/(probabilities.sum(dimensions)+one_hot.sum(dimensions)+1)
    present=one_hot.sum(dimensions)[1:]>0
    dice_loss=(1-dice[1:][present]).mean() if present.any() else logits.sum()*0
    return ce+dice_weight*dice_loss


class Metrics:
    def __init__(self,device):
        self.matrix=torch.zeros((3,3),dtype=torch.float64,device=device)
        # matched prediction, predicted pixels, matched target, target pixels
        self.relaxed=torch.zeros((2,4),dtype=torch.float64,device=device)

    def update(self,logits,target):
        predicted=logits.argmax(1);valid=target!=255
        self.matrix+=torch.bincount((target[valid]*3+predicted[valid]).flatten(),minlength=9).reshape(3,3)
        for row,label in enumerate((1,2)):
            a=(predicted==label)&valid;b=(target==label)&valid
            near_b=F.max_pool2d(b[:,None].float(),3,1,1)[:,0]>0
            near_a=F.max_pool2d(a[:,None].float(),3,1,1)[:,0]>0
            self.relaxed[row]+=torch.stack([(a&near_b).sum(),a.sum(),(b&near_a).sum(),b.sum()])

    def compute(self):
        matrix=self.matrix.cpu();report={}
        for index,name in enumerate(NAMES):
            tp=matrix[index,index].item();pred=matrix[:,index].sum().item();truth=matrix[index].sum().item()
            union=pred+truth-tp
            report[name]={'iou':tp/union if union else None,'dice':2*tp/(pred+truth) if pred+truth else None,
                          'precision':tp/pred if pred else 0.,'recall':tp/truth if truth else 0.,'target_pixels':int(truth)}
            if index:
                matched_p,p,matched_t,t=self.relaxed[index-1].cpu().tolist()
                precision=matched_p/p if p else 0.;recall=matched_t/t if t else 0.
                report[name]['f1_tolerance_1px']=2*precision*recall/(precision+recall) if precision+recall else 0.
        values=[report[name]['iou'] for name in NAMES[1:] if report[name]['iou'] is not None]
        report['foreground_miou']=sum(values)/len(values) if values else 0.
        report['confusion_matrix']=matrix.long().tolist()
        return report
