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
    assert report['foreground_miou']==1 and report['scale_tick']['f1_tolerance_1px']==1
    print('PASS perfect prediction metrics and ignore=255',flush=True)
    dataset=SegmentationDataset('/dataset','train',128,augment=True)
    for index in range(3):
        image,label=dataset[index]
        assert image.shape==(3,128,128) and label.shape==(128,128)
        assert set(label.unique().tolist())<={0,1,2,255}
        assert torch.isfinite(image).all()
    print('PASS real data loading, letterbox and synchronized augmentation',flush=True)


if __name__=='__main__':main()
