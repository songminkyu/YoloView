from yolocode.YOLOv8Thread import YOLOv8Thread
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class YOLOv8PoseThread(YOLOv8Thread):

    def __init__(self):
        super(YOLOv8PoseThread, self).__init__()
        self.task = 'pose'
        self.project = 'runs/pose'

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            classes=self.classes,
            nc=len(self.model.names),
        )

        # 필터링된 preds 및 필터링 여부 리스트 가져오기
        preds, has_filtered = self.filter_and_sort_preds(preds, self.categories, epsilon=1e-5)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, filtered) in enumerate(zip(preds, has_filtered)):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if len(self.categories) == 0 or (filtered and pred is not None):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
                pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
                pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            else:
                pred_kpts = None

            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results
