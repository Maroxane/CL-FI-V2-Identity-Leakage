"""PyTorch Emo-AffectNet static backbone (Ryumina et al., Neurocomputing 2022).

Re-implementation of the VGGFace2-style ResNet-50 used by ElenaRyumina/face_emotion_recognition, whose
FER_static_ResNet50_AffectNet.pt is a plain state_dict (not TorchScript). Keras-like details are kept:
stride on the 1x1 reduce conv, TF 'same' padding, BatchNorm eps=1e-3. Input: BGR, 224x224, minus
(91.4953, 103.8827, 131.0912). extract_features() returns the 512-d fc1 embedding.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn(c):
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class Conv2dSame(nn.Conv2d):
    """TensorFlow 'SAME' padding (asymmetric when needed)."""

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        ph = max((math.ceil(ih / sh) - 1) * sh + kh - ih, 0)
        pw = max((math.ceil(iw / sw) - 1) * sw + kw - iw, 0)
        if ph or pw:
            x = F.pad(x, [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, c, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, 1, stride=stride, bias=False)
        self.batch_norm1 = _bn(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding="same", bias=False)
        self.batch_norm2 = _bn(c)
        self.conv3 = nn.Conv2d(c, c * 4, 1, bias=False)
        self.batch_norm3 = _bn(c * 4)
        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        idt = x if self.i_downsample is None else self.i_downsample(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        return self.relu(x + idt)


class EmoResNet50(nn.Module):
    def __init__(self, num_classes=7, layers=(3, 4, 6, 3)):
        super().__init__()
        self.cin = 64
        self.conv_layer_s2_same = Conv2dSame(3, 64, 7, stride=2, bias=False)
        self.batch_norm1 = _bn(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.layer1 = self._make(layers[0], 64, 1)
        self.layer2 = self._make(layers[1], 128, 2)
        self.layer3 = self._make(layers[2], 256, 2)
        self.layer4 = self._make(layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def _make(self, n, c, stride):
        ds = nn.Sequential(nn.Conv2d(self.cin, c * 4, 1, stride=stride, bias=False), _bn(c * 4))
        blocks = [Bottleneck(self.cin, c, ds, stride)]
        self.cin = c * 4
        blocks += [Bottleneck(self.cin, c) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def extract_features(self, x):
        x = self.max_pool(self.relu(self.batch_norm1(self.conv_layer_s2_same(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc1(self.avgpool(x).flatten(1))

    def forward(self, x):
        return self.fc2(self.relu1(self.extract_features(x)))


def load_emo_affectnet(path, device="cpu"):
    """Loads either the TorchScript export or the state_dict release; raises if keys do not match."""
    try:
        return torch.jit.load(path, map_location=device).eval()
    except RuntimeError:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        m = EmoResNet50(num_classes=sd["fc2.weight"].shape[0])
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval()
