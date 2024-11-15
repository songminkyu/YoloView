from yolocode.YOLOv8Thread import YOLOv8Thread
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class YOLOv8SegThread(YOLOv8Thread):

    def __init__(self):
        super(YOLOv8SegThread, self).__init__()
        self.data = 'YoloView/ultralytics/cfg/datasets/coco128-seg.yaml'  # data_dict
        self.task = 'segment'
        self.project = 'runs/segment'

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=len(self.model.names),
            classes=self.classes,
        )

        p, has_filtered = self.filter_and_sort_preds(p, self.categories, epsilon=1e-5)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, filtered) in enumerate(zip(p, has_filtered)):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]

            if len(self.categories) == 0 or (filtered and pred is not None):
                # categories가 비어 있거나 필터링된 결과가 있는 경우: 원본 pred 사용
                if not len(pred):  # save empty boxes
                    masks = None
                else:
                    masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
            else:
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=None, masks=None))

        return results
