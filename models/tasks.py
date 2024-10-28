import logging
import sys
from utils import glo

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

yoloname = glo.get_value('yoloname')
yoloname1 = glo.get_value('yoloname1')
yoloname2 = glo.get_value('yoloname2')

yolo_name = ((str(yoloname1) if yoloname1 else '') + (str(yoloname2) if str(
    yoloname2) else '')) if yoloname1 or yoloname2 else yoloname