import cv2
from yolocode.YOLOBaseThread import YOLOBaseThread
from pathlib import Path
from glmocr.api import GlmOcr

class GlmOCRSIMPLEThread(YOLOBaseThread):
    def __init__(self):
        super(GlmOCRSIMPLEThread, self).__init__()
        self.task = 'glm-ocr'
        self.project = 'runs/glmocr' # Changed from 'runs/glm-ocr' per user request
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None

    def postprocess(self, lang, img, orig_imgs):
        source = self.source if isinstance(self.source, list) else [self.source]
        import os
        import glob
        
        # Create output directory if it doesn't exist
        os.makedirs(self.project, exist_ok=True)
        
        with GlmOcr() as parser:
            for image_file in source:
                image = cv2.imread(image_file)
                if image is None:
                    continue
                    
                # 원본 이미지 전송
                self.send_input.emit(image)
                
                try:
                    # GLM-OCR parsing (this automatically generates layout_vis temp files if configured)
                    result = parser.parse(str(image_file))
                    
                    # Save the JSON, Markdown, and layout_vis to runs/glmocr as requested
                    result.save(output_dir=self.project)
                    
                    # Read the generated layout visualization image to send to the UI
                    base_name = os.path.basename(image_file)
                    stem = os.path.splitext(base_name)[0]
                    target_dir = os.path.join(self.project, stem, "layout_vis")
                    
                    # Search for the visualization files (e.g., test.jpg or test_page0.jpg)
                    vis_files = glob.glob(os.path.join(target_dir, "*.*"))
                    if vis_files:
                        roi_image = cv2.imread(vis_files[0])
                        if roi_image is not None:
                            self.send_output.emit(roi_image)
                        else:
                            self.send_output.emit(image)
                    else:
                        self.send_output.emit(image)
                    
                except Exception as e:
                    #print(f"Failed: {image_file}: {e}")
                    continue


class GlmOCRThread(YOLOBaseThread):
    def __init__(self):
        super(GlmOCRThread, self).__init__()
        self.task = 'glm-ocr'
        self.project = 'runs/glmocr'  # Changed from 'runs/glm-ocr' per user request
        self.labels_path = None  # 라벨 파일 경로
        self.save_res = None
        self.save_path = None

    def postprocess(self, lang, img, orig_imgs):
        source = self.source if isinstance(self.source, list) else [self.source]
        import os

        # Load yaml to get labels info or customize based on config.yaml (Though parser itself handles labels string representation, this is useful references if needed)
        # We can draw the labels directly via standard cv2.

        # Create output directory if it doesn't exist
        os.makedirs(self.project, exist_ok=True)

        with GlmOcr() as parser:
            for image_file in source:
                image = cv2.imread(image_file)
                if image is None:
                    continue

                # 원본 이미지 전송
                self.send_input.emit(image)

                try:
                    result = parser.parse(str(image_file))

                    # Create a copy for drawing the regions
                    roi_image = image.copy()
                    height, width, _ = roi_image.shape

                    # result.json_result format is [[{index, label, content, bbox_2d}, ...], ...]
                    if hasattr(result, 'json_result') and result.json_result:
                        for page in result.json_result:
                            for region in page:
                                label_name = region.get("label", "Unknown")
                                bbox = region.get("bbox_2d")
                                if bbox and len(bbox) == 4:
                                    # Coordinates are normalized 0-1000, un-normalize them
                                    # x1 = int(nx1 * width / 1000)
                                    nx1, ny1, nx2, ny2 = bbox
                                    x1 = int(nx1 * width / 1000)
                                    y1 = int(ny1 * height / 1000)
                                    x2 = int(nx2 * width / 1000)
                                    y2 = int(ny2 * height / 1000)

                                    # Draw bounding box
                                    color = (0, 255, 0)  # Green box
                                    cv2.rectangle(roi_image, (x1, y1), (x2, y2), color, 2)

                                    # Draw label text
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.5
                                    thickness = 1
                                    # Calculate text size for background box
                                    (text_width, text_height), _ = cv2.getTextSize(label_name, font, font_scale,
                                                                                   thickness)
                                    cv2.rectangle(roi_image, (x1, max(0, y1 - text_height - 5)), (x1 + text_width, y1),
                                                  color, -1)
                                    cv2.putText(roi_image, label_name, (x1, max(0, y1 - 5)), font, font_scale,
                                                (0, 0, 0), thickness)

                    # 결과 이미지 전송
                    self.send_output.emit(roi_image)

                    # Save the JSON and Markdown structured files via the API
                    result.save(output_dir=self.project)

                    # Save the resulting image to the same folder generated by result.save()
                    base_name = os.path.basename(image_file)
                    stem = os.path.splitext(base_name)[0]
                    target_dir = os.path.join(self.project, stem)
                    os.makedirs(target_dir, exist_ok=True)
                    save_file_path = os.path.join(target_dir, base_name)
                    cv2.imwrite(save_file_path, roi_image)

                except Exception as e:
                    # print(f"Failed: {image_file}: {e}")
                    continue
