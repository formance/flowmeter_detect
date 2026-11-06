"""Binary DeepLabV3+ with a ResNet-18 encoder for gauge needles."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def conv_block(in_channels, out_channels, kernel_size, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size // 2),
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class NeedleDeepLabV3Plus(nn.Module):
    def __init__(self, encoder_weights=None):
        super().__init__()
        backbone = resnet18(weights=None)
        if encoder_weights is not None:
            backbone.load_state_dict(torch.load(encoder_weights, map_location='cpu', weights_only=True))

        # Keep the last stage at output stride 16 while retaining pretrained kernels.
        first, second = backbone.layer4
        first.conv1.stride = first.downsample[0].stride = (1, 1)
        for conv in (first.conv2, second.conv1, second.conv2):
            conv.dilation = conv.padding = (2, 2)

        self.encoder = nn.ModuleDict({
            'stem': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
            'low': backbone.layer1,
            'high': nn.Sequential(backbone.layer2, backbone.layer3, backbone.layer4),
        })
        self.aspp = nn.ModuleList([
            conv_block(512, 64, 1),
            *(conv_block(512, 64, 3, dilation=rate) for rate in (6, 12, 18)),
        ])
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(512, 64, 1), nn.ReLU(inplace=True),
        )
        self.project_context = conv_block(320, 64, 1)
        self.project_low = conv_block(64, 32, 1)
        self.classifier = nn.Sequential(
            conv_block(96, 64, 3), conv_block(64, 64, 3), nn.Conv2d(64, 1, 1),
        )

    def forward(self, image):
        low = self.encoder['low'](self.encoder['stem'](image))
        high = self.encoder['high'](low)
        pooled = F.interpolate(self.global_context(high), size=high.shape[-2:], mode='bilinear',
                               align_corners=False)
        context = self.project_context(torch.cat([*(branch(high) for branch in self.aspp), pooled], dim=1))
        context = F.interpolate(context, size=low.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.classifier(torch.cat([context, self.project_low(low)], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
