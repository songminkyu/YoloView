from yolocode.YOLOBaseThread import YOLOBaseThread
from ultralytics.engine.results import Results
from ultralytics.utils import ops, nms


class YOLOPoseThread(YOLOBaseThread):

    def __init__(self):
        super(YOLOPoseThread, self).__init__()
        self.task = 'pose'
        self.project = 'runs/pose'

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        p = super().non_max_suppression(preds)

        # 필터링된 preds 및 필터링 여부 리스트 가져오기
        p, has_filtered = self.filter_and_sort_preds(p, self.categories, epsilon=1e-5)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, (pred, filtered) in enumerate(zip(p, has_filtered)):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if len(self.categories) == 0 or (filtered and pred is not None):
                if pred is None or not len(pred):
                    pred_kpts = None
                else:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    # Use exactly nk columns for keypoints to avoid RuntimeError with extra channels (like DFL)
                    pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
                    # Scale keypoints coordinates to match the original image dimensions
                    pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            else:
                pred_kpts = None

            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6] if pred is not None else None, keypoints=pred_kpts)
            )
        return results
