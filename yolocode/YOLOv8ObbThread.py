import torch
from yolocode.YOLOv8Thread import YOLOv8Thread
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class YOLOv8ObbThread(YOLOv8Thread):

    def __init__(self):
        super(YOLOv8ObbThread, self).__init__()
        self.task = 'obb'
        self.project = 'runs/obb'

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
            nc=len(self.model.names),
            classes=self.classes,
            rotated=True,
        )

        # 필터링된 preds 및 필터링 여부 리스트 가져오기
        preds, has_filtered = self.filter_and_sort_preds(preds, self.categories, epsilon=1e-5)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, filtered, orig_img, img_path in zip(preds, has_filtered, orig_imgs,self.batch[0]):
            if len(self.categories) == 0 or (filtered and pred is not None):
                # 필터링된 결과가 있는 경우: 필터링된 preds 사용
                rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
                rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
                obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            else:
                # 필터링된 결과가 없는 경우: 원본 pred 사용
                obb = None

            # Results 객체 생성
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))

        return results
