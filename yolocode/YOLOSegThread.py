from yolocode.YOLOBaseThread import YOLOBaseThread
from ultralytics.engine.results import Results
from ultralytics.utils import ops, nms

class YOLOSegThread(YOLOBaseThread):

    def __init__(self):
        super(YOLOSegThread, self).__init__()
        self.data = 'YoloView/ultralytics/cfg/datasets/coco128-seg.yaml'  # data_dict
        self.task = 'segment'
        self.project = 'runs/segment'
        self.compile = False
        self.imgsz = (640, 640)

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        proto_preds = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        nms_inputs = preds[0][0] if isinstance(preds[0], tuple) else preds[0]

        p = nms.non_max_suppression(
            nms_inputs,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=len(self.model.names),
            classes=self.classes,
            end2end=getattr(self.model, 'end2end', False),
        )
        p, has_filtered = self.filter_and_sort_preds(p, self.categories, epsilon=1e-5)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []

        for i, (pred, filtered) in enumerate(zip(p, has_filtered)):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]

            if len(self.categories) == 0 or (filtered and pred is not None):
                # categories가 비어 있거나 필터링된 결과가 있는 경우: 원본 pred 사용
                if pred is None or not len(pred):  # save empty boxes
                    masks = None
                else:
                    # Use exactly nm columns for mask coefficients to avoid RuntimeError with extra channels
                    masks = ops.process_mask(proto_preds[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC

                    if masks is not None:
                        # only keep predictions with masks
                        keep = masks.amax((-2, -1)) > 0
                        if not all(keep):
                            pred, masks = pred[keep], masks[keep]

                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
            else:
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=None, masks=None))

        return results
