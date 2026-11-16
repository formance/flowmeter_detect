"""DeepLabV3+ with a pretrained-compatible ResNet encoder and three-class head."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,resnet34,ResNet18_Weights,ResNet34_Weights


def conv(in_channels,out_channels,kernel=3,dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel,padding=dilation*(kernel//2),
                                   dilation=dilation,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))


def dilate(stage,rate):
    for module in stage.modules():
        if isinstance(module,nn.Conv2d):
            if module.stride==(2,2):module.stride=(1,1)
            if module.kernel_size==(3,3):module.dilation=module.padding=(rate,rate)


class DeepLabV3Plus(nn.Module):
    def __init__(self,backbone='resnet18',output_stride=8,decoder_channels=128,encoder_weights='none'):
        super().__init__()
        if backbone not in ('resnet18','resnet34') or output_stride not in (8,16):
            raise ValueError('Use resnet18/resnet34 and output stride 8/16')
        factory,weights_enum=(resnet18,ResNet18_Weights) if backbone=='resnet18' else (resnet34,ResNet34_Weights)
        network=factory(weights=weights_enum.DEFAULT if str(encoder_weights)=='imagenet' else None)
        if str(encoder_weights) not in ('none','imagenet'):
            network.load_state_dict(torch.load(encoder_weights,map_location='cpu',weights_only=True))
        if output_stride==8:dilate(network.layer3,2)
        dilate(network.layer4,4 if output_stride==8 else 2)
        self.encoder=nn.ModuleDict({'stem':nn.Sequential(network.conv1,network.bn1,network.relu,network.maxpool),
                                   'low':network.layer1,'high':nn.Sequential(network.layer2,network.layer3,network.layer4)})
        rates=(12,24,36) if output_stride==8 else (6,12,18)
        self.aspp=nn.ModuleList([conv(512,decoder_channels,1),*[conv(512,decoder_channels,3,rate) for rate in rates]])
        self.pool=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(512,decoder_channels,1),nn.ReLU(inplace=True))
        self.project=nn.Sequential(conv(5*decoder_channels,decoder_channels,1),nn.Dropout(.1))
        self.low_project=conv(64,48,1)
        self.decoder=nn.Sequential(conv(decoder_channels+48,decoder_channels),conv(decoder_channels,decoder_channels),
                                   nn.Conv2d(decoder_channels,3,1))
        self.model_config={'backbone':backbone,'output_stride':output_stride,'decoder_channels':decoder_channels}

    def forward(self,image):
        low=self.encoder['low'](self.encoder['stem'](image));high=self.encoder['high'](low)
        context=F.interpolate(self.pool(high),size=high.shape[-2:],mode='bilinear',align_corners=False)
        high=self.project(torch.cat([*[branch(high) for branch in self.aspp],context],dim=1))
        high=F.interpolate(high,size=low.shape[-2:],mode='bilinear',align_corners=False)
        logits=self.decoder(torch.cat([high,self.low_project(low)],dim=1))
        return F.interpolate(logits,size=image.shape[-2:],mode='bilinear',align_corners=False)


def load_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('architecture')!='deeplabv3plus_pointer_scale' or checkpoint.get('classes')!=['background','pointer','scale_tick']:
        raise ValueError('Expected a three-class pointer/scale DeepLabV3+ checkpoint')
    model=DeepLabV3Plus(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device),checkpoint
