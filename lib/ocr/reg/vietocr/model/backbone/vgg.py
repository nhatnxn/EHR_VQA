import torch
import torch.nn as nn 
from torchvision import models

class Vgg(nn.Module):
    def __init__(self, name, ss, ks, hidden, pretrained=True, dropout=0.5):
        super(Vgg, self).__init__()

        cnn = models.vgg19_bn(pretrained=pretrained)

        pool_idx = 0

        for i, layer in enumerate(cnn.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                cnn.features[i] = torch.nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0)
                pool_idx += 1

        self.features = cnn.features
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(512, hidden, 1)

    def forward(self, x):
        """
        Shape:
            - x: (N, C, H, W)
            - output: (W, N, C)
        """

        conv = self.features(x)
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)  # B*C*H*W
        # print(conv.shape)
        # from IPython import embed; embed()

        #        conv = rearrange(conv, 'b d h w -> b d (w h)')
        conv = conv.transpose(-1, -2)  # B*C*W*H
        conv = conv.flatten(2)  # B*C* (WxH)
        conv = conv.permute(2, 0, 1)  # T*B*C

        return conv


def vgg19_bn(ss, ks, hidden, pretrained=True, dropout=0.5):
    return Vgg('vgg19_bn', ss, ks, hidden, pretrained, dropout)


