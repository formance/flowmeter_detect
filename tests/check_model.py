"""CPU checks for segmentation output, losses, training gradients and metrics."""
import torch

from src.common.dataset import SegmentationDataset
from src.common.metrics import Metrics,segmentation_loss
from src.common.model import DeepLabV3Plus


def main():
    torch.set_num_threads(2);torch.manual_seed(19)
    target=torch.zeros(2,64,64,dtype=torch.long)
    target[:,10:50,29:33]=1;target[:,15:45:5,8:22]=2;target[:,:2]=255
    for backbone,stride in [('resnet18',8),('resnet18',16),('resnet34',8)]:
        model=DeepLabV3Plus(backbone,stride,32)
        model.train();logits=model(torch.randn(2,3,64,64))
        assert logits.shape==(2,3,64,64)
        loss=segmentation_loss(logits,target,torch.tensor([.2,1.,2.]))
        loss.backward()
        assert torch.isfinite(loss) and model.decoder[-1].weight.grad.abs().sum()>0
        print('PASS forward/backward',backbone,'OS',stride,flush=True)
    perfect=torch.nn.functional.one_hot(target.clamp(0,2),3).permute(0,3,1,2).float()*20
    metrics=Metrics('cpu');metrics.update(perfect,target);report=metrics.compute()
    assert report['foreground_mean_recall']==1 and report['scale_tick']['recall']==1
    print('PASS perfect prediction metrics and ignore=255',flush=True)
    extra=perfect.clone();extra[:,2,50:60,10:20]=40;extra[:,1,:2]=40
    changed=Metrics('cpu');changed.update(extra,target)
    assert changed.compute()==report
    missed=perfect.clone();missed[:,2,15,8:22]=-40
    check=Metrics('cpu');check.update(missed,target);partial=check.compute()
    assert partial['pointer']['recall']==1 and abs(partial['scale_tick']['recall']-5/6)<1e-12
    assert abs(partial['foreground_mean_recall']-11/12)<1e-12
    absent=Metrics('cpu');absent.update(torch.zeros(1,3,4,4),torch.zeros(1,4,4,dtype=torch.long))
    assert absent.compute()['scale_tick']['recall'] is None
    assert set(report)=={'pointer','scale_tick','foreground_mean_recall'}
    print('PASS GT-only recall: false positives/padding ignored, missing GT pixels counted, class macro average',flush=True)
    weights=torch.tensor([.2,1.,2.])
    dice_loss=segmentation_loss(perfect,target,weights)
    balanced=segmentation_loss(perfect,target,weights,scale_loss='tversky',tversky_alpha=.5,tversky_beta=.5)
    assert torch.allclose(dice_loss,balanced,atol=1e-6)
    false_positive=perfect.clone();false_positive[:,2,50:60,10:20]=40
    fp_heavy=segmentation_loss(false_positive,target,weights,scale_loss='tversky',tversky_alpha=.7,tversky_beta=.3)
    fn_heavy=segmentation_loss(false_positive,target,weights,scale_loss='tversky',tversky_alpha=.3,tversky_beta=.7)
    assert fp_heavy>fn_heavy
    false_positive.requires_grad_()
    segmentation_loss(false_positive,target,weights,scale_loss='tversky').backward()
    assert torch.isfinite(false_positive.grad).all()
    print('PASS Tversky matches balanced Dice, penalizes false positives and has finite gradients',flush=True)
    dataset=SegmentationDataset('/dataset','train',128,augment=True)
    for index in range(3):
        image,label=dataset[index]
        assert image.shape==(3,128,128) and label.shape==(128,128)
        assert set(label.unique().tolist())<={0,1,2,255}
        assert torch.isfinite(image).all()
    print('PASS real data loading, letterbox and synchronized augmentation',flush=True)


if __name__=='__main__':main()
