from yolocode.YOLOv8Thread import YOLOv8Thread
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class BBoxValidThread(YOLOv8Thread):
    def __init__(self):
        super(YOLOv8Thread, self).__init__()
        self.task = 'bbox_valid'
        self.project = 'runs/bbox_valid'