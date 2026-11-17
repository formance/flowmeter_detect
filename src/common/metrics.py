"""Segmentation losses and exact recall on foreground ground-truth pixels only."""
import torch
from torch.nn import functional as F

NAMES=('pointer','scale_tick')


def segmentation_loss(logits,target,class_weights,dice_weight=1.,scale_loss='dice',tversky_alpha=.7,tversky_beta=.3):
    logits=logits.float();valid=target!=255
    ce=F.cross_entropy(logits,target,weight=class_weights,ignore_index=255)
    one_hot=F.one_hot(target.clamp(0,2),3).permute(0,3,1,2).float()*valid[:,None]
    probabilities=logits.softmax(1)*valid[:,None]
    dimensions=(0,2,3)
    intersection=(probabilities*one_hot).sum(dimensions)
    dice=(2*intersection+1)/(probabilities.sum(dimensions)+one_hot.sum(dimensions)+1)
    present=one_hot.sum(dimensions)[1:]>0
    foreground_losses=1-dice[1:]
    if scale_loss=='tversky':
        tp=intersection[2]
        fp=probabilities.sum(dimensions)[2]-tp
        fn=one_hot.sum(dimensions)[2]-tp
        # alpha penalizes false positives, beta penalizes false negatives.
        # Smooth .5 makes alpha=beta=.5 equivalent to the Dice above.
        score=(tp+.5)/(tp+tversky_alpha*fp+tversky_beta*fn+.5)
        foreground_losses=torch.stack((1-dice[1],1-score))
    elif scale_loss!='dice':
        raise ValueError(f'Unknown scale loss: {scale_loss}')
    dice_loss=foreground_losses[present].mean() if present.any() else logits.sum()*0
    return ce+dice_weight*dice_loss


class Metrics:
    def __init__(self,device):
        # Per foreground class: correctly predicted GT pixels, all GT pixels.
        self.counts=torch.zeros((2,2),dtype=torch.int64,device=device)

    def update(self,logits,target):
        predicted=logits.argmax(1)
        for row,label in enumerate((1,2)):
            truth=target==label
            self.counts[row]+=torch.stack(((truth&(predicted==label)).sum(),truth.sum()))

    def compute(self):
        counts=self.counts.cpu().tolist();report={};values=[]
        for index,name in enumerate(NAMES):
            matched,truth=counts[index]
            recall=matched/truth if truth else None
            report[name]={'recall':recall,'matched_pixels':matched,'target_pixels':truth}
            if recall is not None:values.append(recall)
        report['foreground_mean_recall']=sum(values)/len(values) if values else 0.
        return report
