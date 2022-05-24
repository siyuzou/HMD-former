import os
import os.path as osp
import joblib

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

# from core.config import cfg
from core.util.exe_util.logger import logger


class ResNet50(nn.Module):

    def __init__(self, num_scales=1, pretrain_path=None):
        assert num_scales in [1, 2, 3]

        block, layers, name = Bottleneck, [3, 4, 6, 3], 'resnet50'

        if num_scales == 1:
            self.strides = [32]
            self.num_channels = [2048]
        if num_scales == 2:
            self.strides = [16, 32]
            self.num_channels = [1024, 2048]
        elif num_scales == 3:
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]

        self.name = name
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain_path is not None:
            self.init_weights(pretrain_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)  # (B, 256, 56,56)
        x3 = self.layer2(x2)  # (B, 512, 28, 28)
        x4 = self.layer3(x3)  # (B, 1024, 14, 14)
        x5 = self.layer4(x4)  # (B, 2048, 7, 7)

        # now, return the last feature map before avgpool
        # xf = self.avgpool(x5)   # (B, 2048, 1, 1)
        # xf = xf.view(xf.size(0), -1)    # (B, 2048)
        # return xf

        if self.strides == [32]:
            return [x5]
        elif self.strides == [16, 32]:
            return [x4, x5]
        elif self.strides == [8, 16, 32]:
            return [x3, x4, x5]

    def init_weights(self, pretrain_path):
        if pretrain_path is None or not osp.exists(pretrain_path):
            logger.info(f'No ckpt set for resnet50, skip using ckpt ot init.')
            return

        # pretrain_path = osp.join(cfg.NETWORK.backbone.root_dir, 'resnet50-19c8e357.pth')
        org_resnet = torch.load(pretrain_path)
        # org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)

        self.load_state_dict(org_resnet)
        logger.info(f"Initialize resnet from: {pretrain_path}")


if __name__ == '__main__':
    resnet = ResNet50()
    resnet.init_weights()
    # todo: pytorch 的默认数据类型是 float32 ?
    x = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    f = resnet(x)
    print(f.shape)
