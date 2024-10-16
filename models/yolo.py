import argparse
import logging
import sys
import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import math
import torch
import torch.nn as nn
from utils import glo
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

yoloname = glo.get_value('yoloname')
yoloname1 = glo.get_value('yoloname1')
yoloname2 = glo.get_value('yoloname2')

yolo_name = ((str(yoloname1) if yoloname1 else '') + (str(yoloname2) if str(
    yoloname2) else '')) if yoloname1 or yoloname2 else yoloname