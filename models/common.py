import math
import torch
import torch.nn as nn
import warnings

from utils import glo

try:
    import ultralytics
    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os
    os.system("pip install -U ultralytics")
    import ultralytics

yoloname = glo.get_value('yoloname')
yoloname1 = glo.get_value('yoloname1')
yoloname2 = glo.get_value('yoloname2')

yolo_name = ((str(yoloname1) if yoloname1 else '') + (str(yoloname2) if str(
    yoloname2) else '')) if yoloname1 or yoloname2 else yoloname


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
        # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
        default_act = nn.SiLU()  # default activation

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def forward_fuse(self, x):
            return self.act(self.conv(x))
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
class TransformerLayer(nn.Module):
        # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
        def __init__(self, c, num_heads):
            super().__init__()
            self.q = nn.Linear(c, c, bias=False)
            self.k = nn.Linear(c, c, bias=False)
            self.v = nn.Linear(c, c, bias=False)
            self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
            self.fc1 = nn.Linear(c, c, bias=False)
            self.fc2 = nn.Linear(c, c, bias=False)

        def forward(self, x):
            x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
            x = self.fc2(self.fc1(x)) + x
            return x
class TransformerBlock(nn.Module):
        # Vision Transformer https://arxiv.org/abs/2010.11929
        def __init__(self, c1, c2, num_heads, num_layers):
            super().__init__()
            self.conv = None
            if c1 != c2:
                self.conv = Conv(c1, c2)
            self.linear = nn.Linear(c2, c2)  # learnable position embedding
            self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
            self.c2 = c2

        def forward(self, x):
            if self.conv is not None:
                x = self.conv(x)
            b, _, w, h = x.shape
            p = x.flatten(2).permute(2, 0, 1)
            return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
class Bottleneck(nn.Module):
        # Standard bottleneck
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class BottleneckCSP(nn.Module):
        # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
        def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                     e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
            self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
            self.cv4 = Conv(2 * c_, c2, 1, 1)
            self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
            self.act = nn.SiLU()
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            y1 = self.cv3(self.m(self.cv1(x)))
            y2 = self.cv2(x)
            return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
class Contract(nn.Module):
        # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
        def __init__(self, gain=2):
            super().__init__()
            self.gain = gain

        def forward(self, x):
            b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
            s = self.gain
            x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
            return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)
class Expand(nn.Module):
        # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
        def __init__(self, gain=2):
            super().__init__()
            self.gain = gain

        def forward(self, x):
            b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
            s = self.gain
            x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
            x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
            return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)
class Concat(nn.Module):
        # Concatenate a list of tensors along dimension
        def __init__(self, dimension=1):
            super().__init__()
            self.d = dimension

        def forward(self, x):
            return torch.cat(x, self.d)
class Proto(nn.Module):
        # YOLOv5 mask Proto module for segmentation models
        def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
            super().__init__()
            self.cv1 = Conv(c1, c_, k=3)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.cv2 = Conv(c_, c_, k=3)
            self.cv3 = Conv(c_, c2)

        def forward(self, x):
            return self.cv3(self.cv2(self.upsample(self.cv1(x))))
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
