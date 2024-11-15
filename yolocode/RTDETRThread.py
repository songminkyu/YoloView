import numpy as np
import torch
from yolocode.YOLOv8Thread import YOLOv8Thread
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops

class RTDETRThread(YOLOv8Thread):
    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.conf_thres  # (300, )
            if self.classes is not None:
                idx = (cls == torch.tensor(self.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            preds, has_filtered = self.filter_and_sort_preds([pred], self.categories, epsilon=1e-5)
            # filter_and_sort_preds returns in list form, so access the first element.
            pred = preds[0]
            filtered = has_filtered[0]

            if len(self.categories) == 0 or (filtered and pred is not None):
                oh, ow = orig_img.shape[:2]
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            else:
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=None))

        return results

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]
