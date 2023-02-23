import torch
import torch.nn as nn
import timm

class NetworkB3(nn.Module):
    def __init__(self):
        super(NetworkB3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x

class NetworkWrn50(nn.Module):
    def __init__(self):
        super(NetworkWrn50, self).__init__()
        self.model = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x

class NetworkWrn101(nn.Module):
    def __init__(self):
        super(NetworkWrn101, self).__init__()
        self.model = timm.create_model('wide_resnet101_2', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x

class NetworkCait(nn.Module):
    def __init__(self):
        super(NetworkCait, self).__init__()
        self.model = timm.create_model('cait_s36_384', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x

class NetworkB7(nn.Module):
    def __init__(self):
        super(NetworkB7, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x

class NetworkB6(nn.Module):
    def __init__(self):
        super(NetworkB6, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b6', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)

        return x