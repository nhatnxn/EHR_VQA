import torch.nn as nn 
from .vgg import vgg19_bn
class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        self.model = vgg19_bn(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)