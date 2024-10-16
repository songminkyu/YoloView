import ast
import contextlib
import importlib
import json
import math
import numpy as np
import os
import pandas as pd
import platform
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from PIL import Image
from torch.cuda import amp
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d
from urllib.parse import urlparse
from typing import Optional
from IPython.display import display
import cv2
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

### --- YOLOv8 Code --- ###
if "yolov8" in yolo_name or "rtdetr" in yolo_name or "yolov10" in yolo_name or "yolo11" in yolo_name:
    from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, ROOT, yaml_load
    from ultralytics.utils.checks import check_requirements, check_suffix, check_version, check_yaml
    from ultralytics.utils.downloads import attempt_download_asset, is_url
    def check_class_names(names):
        """
        Check class names.

        Map imagenet class codes to human-readable names if required. Convert lists to dicts.
        """
        if isinstance(names, list):  # names is a list
            names = dict(enumerate(names))  # convert to dict
        if isinstance(names, dict):
            # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
            names = {int(k): str(v) for k, v in names.items()}
            n = len(names)
            if max(names.keys()) >= n:
                raise KeyError(
                    f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                    f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
                )
            if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
                names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
                names = {k: names_map[v] for k, v in names.items()}
        return names
    def default_class_names(data=None):
        """Applies default class names to an input YAML file or returns numerical class names."""
        if data:
            with contextlib.suppress(Exception):
                return yaml_load(check_yaml(data))["names"]
        return {i: f"class{i}" for i in range(999)}  # return default if above errors
    class AutoBackend(nn.Module):
        """
        Handles dynamic backend selection for running inference using Ultralytics YOLO models.

        The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
        range of formats, each with specific naming conventions as outlined below:

            Supported Formats and Naming Conventions:
                | Format                | File Suffix      |
                |-----------------------|------------------|
                | PyTorch               | *.pt             |
                | TorchScript           | *.torchscript    |
                | ONNX Runtime          | *.onnx           |
                | ONNX OpenCV DNN       | *.onnx (dnn=True)|
                | OpenVINO              | *openvino_model/ |
                | CoreML                | *.mlpackage      |
                | TensorRT              | *.engine         |
                | TensorFlow SavedModel | *_saved_model    |
                | TensorFlow GraphDef   | *.pb             |
                | TensorFlow Lite       | *.tflite         |
                | TensorFlow Edge TPU   | *_edgetpu.tflite |
                | PaddlePaddle          | *_paddle_model   |
                | NCNN                  | *_ncnn_model     |

        This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
        models across various platforms.
        """

        @torch.no_grad()
        def __init__(
                self,
                weights="yolov8n.pt",
                device=torch.device("cpu"),
                dnn=False,
                data=None,
                fp16=False,
                batch=1,
                fuse=True,
                verbose=True,
        ):
            """
            Initialize the AutoBackend for inference.

            Args:
                weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
                device (torch.device): Device to run the model on. Defaults to CPU.
                dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
                data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
                fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
                batch (int): Batch-size to assume for inference.
                fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
                verbose (bool): Enable verbose logging. Defaults to True.
            """
            super().__init__()
            w = str(weights[0] if isinstance(weights, list) else weights)
            nn_module = isinstance(weights, torch.nn.Module)
            (
                pt,
                jit,
                onnx,
                xml,
                engine,
                coreml,
                saved_model,
                pb,
                tflite,
                edgetpu,
                tfjs,
                paddle,
                ncnn,
                triton,
            ) = self._model_type(w)
            fp16 &= pt or jit or onnx or xml or engine or nn_module or triton  # FP16
            nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
            stride = 32  # default stride
            model, metadata = None, None

            # Set device
            cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
            if cuda and not any([nn_module, pt, jit, engine, onnx]):  # GPU dataloader formats
                device = torch.device("cpu")
                cuda = False

            # Download if not local
            if not (pt or triton or nn_module):
                w = attempt_download_asset(w)

            # In-memory PyTorch model
            if nn_module:
                model = weights.to(device)
                if fuse:
                    model = model.fuse(verbose=verbose)
                if hasattr(model, "kpt_shape"):
                    kpt_shape = model.kpt_shape  # pose-only
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, "module") else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
                pt = True

            # PyTorch
            elif pt:
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(
                    weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
                )
                if hasattr(model, "kpt_shape"):
                    kpt_shape = model.kpt_shape  # pose-only
                stride = max(int(model.stride.max()), 32)  # model stride
                names = model.module.names if hasattr(model, "module") else model.names  # get class names
                model.half() if fp16 else model.float()
                self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

            # TorchScript
            elif jit:
                LOGGER.info(f"Loading {w} for TorchScript inference...")
                extra_files = {"config.txt": ""}  # model metadata
                model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
                model.half() if fp16 else model.float()
                if extra_files["config.txt"]:  # load metadata dict
                    metadata = json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items()))

            # ONNX OpenCV DNN
            elif dnn:
                LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
                check_requirements("opencv-python>=4.5.4")
                net = cv2.dnn.readNetFromONNX(w)

            # ONNX Runtime
            elif onnx:
                LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
                check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
                if IS_RASPBERRYPI or IS_JETSON:
                    # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                    check_requirements("numpy==1.23.5")
                import onnxruntime

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
                session = onnxruntime.InferenceSession(w, providers=providers)
                output_names = [x.name for x in session.get_outputs()]
                metadata = session.get_modelmeta().custom_metadata_map

            # OpenVINO
            elif xml:
                LOGGER.info(f"Loading {w} for OpenVINO inference...")
                check_requirements("openvino>=2024.0.0")
                import openvino as ov

                core = ov.Core()
                w = Path(w)
                if not w.is_file():  # if not *.xml
                    w = next(w.glob("*.xml"))  # get *.xml file from *_openvino_model dir
                ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
                if ov_model.get_parameters()[0].get_layout().empty:
                    ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

                # OpenVINO inference modes are 'LATENCY', 'THROUGHPUT' (not recommended), or 'CUMULATIVE_THROUGHPUT'
                inference_mode = "CUMULATIVE_THROUGHPUT" if batch > 1 else "LATENCY"
                LOGGER.info(f"Using OpenVINO {inference_mode} mode for batch={batch} inference...")
                ov_compiled_model = core.compile_model(
                    ov_model,
                    device_name="AUTO",  # AUTO selects best available device, do not modify
                    config={"PERFORMANCE_HINT": inference_mode},
                )
                input_name = ov_compiled_model.input().get_any_name()
                metadata = w.parent / "metadata.yaml"

            # TensorRT
            elif engine:
                LOGGER.info(f"Loading {w} for TensorRT inference...")
                try:
                    import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
                except ImportError:
                    if LINUX:
                        check_requirements("tensorrt>7.0.0,<=10.1.0")
                    import tensorrt as trt  # noqa
                check_version(trt.__version__, ">=7.0.0", hard=True)
                check_version(trt.__version__, "<=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
                if device.type == "cpu":
                    device = torch.device("cuda:0")
                Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
                logger = trt.Logger(trt.Logger.INFO)
                # Read file
                with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                    try:
                        meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                        metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                    except UnicodeDecodeError:
                        f.seek(0)  # engine file may lack embedded Ultralytics metadata
                    model = runtime.deserialize_cuda_engine(f.read())  # read engine

                # Model context
                try:
                    context = model.create_execution_context()
                except Exception as e:  # model is None
                    LOGGER.error(f"ERROR: TensorRT model exported with a different version than {trt.__version__}\n")
                    raise e

                bindings = OrderedDict()
                output_names = []
                fp16 = False  # default updated below
                dynamic = False
                is_trt10 = not hasattr(model, "num_bindings")
                num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
                for i in num:
                    if is_trt10:
                        name = model.get_tensor_name(i)
                        dtype = trt.nptype(model.get_tensor_dtype(name))
                        is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                        if is_input:
                            if -1 in tuple(model.get_tensor_shape(name)):
                                dynamic = True
                                context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                                if dtype == np.float16:
                                    fp16 = True
                        else:
                            output_names.append(name)
                        shape = tuple(context.get_tensor_shape(name))
                    else:  # TensorRT < 10.0
                        name = model.get_binding_name(i)
                        dtype = trt.nptype(model.get_binding_dtype(i))
                        is_input = model.binding_is_input(i)
                        if model.binding_is_input(i):
                            if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                                dynamic = True
                                context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                            if dtype == np.float16:
                                fp16 = True
                        else:
                            output_names.append(name)
                        shape = tuple(context.get_binding_shape(i))
                    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
                binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
                batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size

            # CoreML
            elif coreml:
                LOGGER.info(f"Loading {w} for CoreML inference...")
                import coremltools as ct

                model = ct.models.MLModel(w)
                metadata = dict(model.user_defined_metadata)

            # TF SavedModel
            elif saved_model:
                LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
                import tensorflow as tf

                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
                metadata = Path(w) / "metadata.yaml"

            # TF GraphDef
            elif pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
                import tensorflow as tf

                from ultralytics.engine.exporter import gd_outputs

                def wrap_frozen_graph(gd, inputs, outputs):
                    """Wrap frozen graphs for deployment."""
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # TF GraphDef
                with open(w, "rb") as f:
                    gd.ParseFromString(f.read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
                with contextlib.suppress(StopIteration):  # find metadata in SavedModel alongside GraphDef
                    metadata = next(Path(w).resolve().parent.rglob(f"{Path(w).stem}_saved_model*/metadata.yaml"))

            # TFLite or TFLite Edge TPU
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf

                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
                if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                    delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                        platform.system()
                    ]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # TFLite
                    LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
                # Load metadata
                with contextlib.suppress(zipfile.BadZipFile):
                    with zipfile.ZipFile(w, "r") as model:
                        meta_file = model.namelist()[0]
                        metadata = ast.literal_eval(model.read(meta_file).decode("utf-8"))

            # TF.js
            elif tfjs:
                raise NotImplementedError("YOLOv8 TF.js inference is not currently supported.")

            # PaddlePaddle
            elif paddle:
                LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
                check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
                import paddle.inference as pdi  # noqa

                w = Path(w)
                if not w.is_file():  # if not *.pdmodel
                    w = next(w.rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
                config = pdi.Config(str(w), str(w.with_suffix(".pdiparams")))
                if cuda:
                    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
                predictor = pdi.create_predictor(config)
                input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
                output_names = predictor.get_output_names()
                metadata = w.parents[1] / "metadata.yaml"

            # NCNN
            elif ncnn:
                LOGGER.info(f"Loading {w} for NCNN inference...")
                check_requirements("git+https://github.com/Tencent/ncnn.git" if ARM64 else "ncnn")  # requires NCNN
                import ncnn as pyncnn

                net = pyncnn.Net()
                net.opt.use_vulkan_compute = cuda
                w = Path(w)
                if not w.is_file():  # if not *.param
                    w = next(w.glob("*.param"))  # get *.param file from *_ncnn_model dir
                net.load_param(str(w))
                net.load_model(str(w.with_suffix(".bin")))
                metadata = w.parent / "metadata.yaml"

            # NVIDIA Triton Inference Server
            elif triton:
                check_requirements("tritonclient[all]")
                from ultralytics.utils.triton import TritonRemoteModel

                model = TritonRemoteModel(w)

            # Any other format (unsupported)
            else:
                from ultralytics.engine.exporter import export_formats

                raise TypeError(
                    f"model='{w}' is not a supported model format. Ultralytics supports: {export_formats()['Format']}\n"
                    f"See https://docs.ultralytics.com/modes/predict for help."
                )

            # Load external metadata YAML
            if isinstance(metadata, (str, Path)) and Path(metadata).exists():
                metadata = yaml_load(metadata)
            if metadata and isinstance(metadata, dict):
                for k, v in metadata.items():
                    if k in {"stride", "batch"}:
                        metadata[k] = int(v)
                    elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                        metadata[k] = eval(v)
                stride = metadata["stride"]
                task = metadata["task"]
                batch = metadata["batch"]
                imgsz = metadata["imgsz"]
                names = metadata["names"]
                kpt_shape = metadata.get("kpt_shape")
            elif not (pt or triton or nn_module):
                LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

            # Check names
            if "names" not in locals():  # names missing
                names = default_class_names(data)
            names = check_class_names(names)

            # Disable gradients
            if pt:
                for p in model.parameters():
                    p.requires_grad = False

            self.__dict__.update(locals())  # assign all variables to self

        def forward(self, im, augment=False, visualize=False, embed=None):
            """
            Runs inference on the YOLOv8 MultiBackend model.

            Args:
                im (torch.Tensor): The image tensor to perform inference on.
                augment (bool): whether to perform data augmentation during inference, defaults to False
                visualize (bool): whether to visualize the output predictions, defaults to False
                embed (list, optional): A list of feature vectors/embeddings to return.

            Returns:
                (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
            """
            b, ch, h, w = im.shape  # batch, channel, height, width
            if self.fp16 and im.dtype != torch.float16:
                im = im.half()  # to FP16
            if self.nhwc:
                im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

            # PyTorch
            if self.pt or self.nn_module:
                y = self.model(im, augment=augment, visualize=visualize, embed=embed)

            # TorchScript
            elif self.jit:
                y = self.model(im)

            # ONNX OpenCV DNN
            elif self.dnn:
                im = im.cpu().numpy()  # torch to numpy
                self.net.setInput(im)
                y = self.net.forward()

            # ONNX Runtime
            elif self.onnx:
                im = im.cpu().numpy()  # torch to numpy
                y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

            # OpenVINO
            elif self.xml:
                im = im.cpu().numpy()  # FP32

                if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:  # optimized for larger batch-sizes
                    n = im.shape[0]  # number of images in batch
                    results = [None] * n  # preallocate list with None to match the number of images

                    def callback(request, userdata):
                        """Places result in preallocated list using userdata index."""
                        results[userdata] = request.results

                    # Create AsyncInferQueue, set the callback and start asynchronous inference for each input image
                    async_queue = self.ov.runtime.AsyncInferQueue(self.ov_compiled_model)
                    async_queue.set_callback(callback)
                    for i in range(n):
                        # Start async inference with userdata=i to specify the position in results list
                        async_queue.start_async(inputs={self.input_name: im[i: i + 1]},
                                                userdata=i)  # keep image as BCHW
                    async_queue.wait_all()  # wait for all inference requests to complete
                    y = np.concatenate([list(r.values())[0] for r in results])

                else:  # inference_mode = "LATENCY", optimized for fastest first result at batch-size 1
                    y = list(self.ov_compiled_model(im).values())

            # TensorRT
            elif self.engine:
                if self.dynamic or im.shape != self.bindings["images"].shape:
                    if self.is_trt10:
                        self.context.set_input_shape("images", im.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                        for name in self.output_names:
                            self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                    else:
                        i = self.model.get_binding_index("images")
                        self.context.set_binding_shape(i, im.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                        for name in self.output_names:
                            i = self.model.get_binding_index(name)
                            self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

                s = self.bindings["images"].shape
                assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs["images"] = int(im.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = [self.bindings[x].data for x in sorted(self.output_names)]

            # CoreML
            elif self.coreml:
                im = im[0].cpu().numpy()
                im_pil = Image.fromarray((im * 255).astype("uint8"))
                # im = im.resize((192, 320), Image.BILINEAR)
                y = self.model.predict({"image": im_pil})  # coordinates are xywh normalized
                if "confidence" in y:
                    raise TypeError(
                        "Ultralytics only supports inference of non-pipelined CoreML models exported with "
                        f"'nms=False', but 'model={w}' has an NMS pipeline created by an 'nms=True' export."
                    )
                    # TODO: CoreML NMS inference handling
                    # from ultralytics.utils.ops import xywh2xyxy
                    # box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                    # conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float32)
                    # y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
                elif len(y) == 1:  # classification model
                    y = list(y.values())
                elif len(y) == 2:  # segmentation model
                    y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)

            # PaddlePaddle
            elif self.paddle:
                im = im.cpu().numpy().astype(np.float32)
                self.input_handle.copy_from_cpu(im)
                self.predictor.run()
                y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]

            # NCNN
            elif self.ncnn:
                mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
                with self.net.create_extractor() as ex:
                    ex.input(self.net.input_names()[0], mat_in)
                    # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
                    y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]

            # NVIDIA Triton Inference Server
            elif self.triton:
                im = im.cpu().numpy()  # torch to numpy
                y = self.model(im)

            # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            else:
                im = im.cpu().numpy()
                if self.saved_model:  # SavedModel
                    y = self.model(im, training=False) if self.keras else self.model(im)
                    if not isinstance(y, list):
                        y = [y]
                elif self.pb:  # GraphDef
                    y = self.frozen_func(x=self.tf.constant(im))
                else:  # Lite or Edge TPU
                    details = self.input_details[0]
                    is_int = details["dtype"] in {np.int8, np.int16}  # is TFLite quantized int8 or int16 model
                    if is_int:
                        scale, zero_point = details["quantization"]
                        im = (im / scale + zero_point).astype(details["dtype"])  # de-scale
                    self.interpreter.set_tensor(details["index"], im)
                    self.interpreter.invoke()
                    y = []
                    for output in self.output_details:
                        x = self.interpreter.get_tensor(output["index"])
                        if is_int:
                            scale, zero_point = output["quantization"]
                            x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                        if x.ndim == 3:  # if task is not classification, excluding masks (ndim=4) as well
                            # Denormalize xywh by image size. See https://github.com/ultralytics/ultralytics/pull/1695
                            # xywh are normalized in TFLite/EdgeTPU to mitigate quantization error of integer models
                            if x.shape[-1] == 6:  # end-to-end model
                                x[:, :, [0, 2]] *= w
                                x[:, :, [1, 3]] *= h
                            else:
                                x[:, [0, 2]] *= w
                                x[:, [1, 3]] *= h
                        y.append(x)
                # TF segment fixes: export is reversed vs ONNX export and protos are transposed
                if len(y) == 2:  # segment with (det, proto) output order reversed
                    if len(y[1].shape) != 4:
                        y = list(reversed(y))  # should be y = (1, 116, 8400), (1, 160, 160, 32)
                    if y[1].shape[-1] == 6:  # end-to-end model
                        y = [y[1]]
                    else:
                        y[1] = np.transpose(y[1], (0, 3, 1, 2))  # should be y = (1, 116, 8400), (1, 32, 160, 160)
                y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

            # for x in y:
            #     print(type(x), len(x)) if isinstance(x, (list, tuple)) else print(type(x), x.shape)  # debug shapes
            if isinstance(y, (list, tuple)):
                if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                    ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                    nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                    self.names = {i: f"class{i}" for i in range(nc)}
                return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
            else:
                return self.from_numpy(y)

        def from_numpy(self, x):
            """
            Convert a numpy array to a tensor.

            Args:
                x (np.ndarray): The array to be converted.

            Returns:
                (torch.Tensor): The converted tensor
            """
            return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

        def warmup(self, imgsz=(1, 3, 640, 640)):
            """
            Warm up the model by running one forward pass with a dummy input.

            Args:
                imgsz (tuple): The shape of the dummy input tensor in the format (batch_size, channels, height, width)
            """
            import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

            warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton, self.nn_module
            if any(warmup_types) and (self.device.type != "cpu" or self.triton):
                im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):
                    self.forward(im)  # warmup

        @staticmethod
        def _model_type(p="path/to/model.pt"):
            """
            Takes a path to a model file and returns the model type. Possibles types are pt, jit, onnx, xml, engine, coreml,
            saved_model, pb, tflite, edgetpu, tfjs, ncnn or paddle.

            Args:
                p: path to the model file. Defaults to path/to/model.pt

            Examples:
                >>> model = AutoBackend(weights="path/to/model.onnx")
                >>> model_type = model._model_type()  # returns "onnx"
            """
            from ultralytics.engine.exporter import export_formats

            sf = export_formats()["Suffix"]  # export suffixes
            if not is_url(p) and not isinstance(p, str):
                check_suffix(p, sf)  # checks
            name = Path(p).name
            types = [s in name for s in sf]
            types[5] |= name.endswith(".mlmodel")  # retain support for older Apple CoreML *.mlmodel formats
            types[8] &= not types[9]  # tflite &= not edgetpu
            if any(types):
                triton = False
            else:
                from urllib.parse import urlsplit

                url = urlsplit(p)
                triton = bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}

            return types + [triton]
### --- YOLOv8 Code --- ###

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
