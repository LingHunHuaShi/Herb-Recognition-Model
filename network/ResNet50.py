from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet, self).__init__()
        # if pretrained:
        #     self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # else:
        #     self.resnet = models.resnet50()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        return self.resnet(x)

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    model = ResNet(20, pretrained=True)
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print('ResNet parameters: {:.2f} millions.'.format(num_params / 1E6))

