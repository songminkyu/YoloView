import os.path
import time

import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from pathlib import Path

from models.common import DetectMultiBackend_YOLOv9
from yolocode.yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolocode.yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow,
                                           check_requirements, colorstr, cv2,
                                           increment_path, non_max_suppression, print_args, scale_boxes,
                                           strip_optimizer, xyxy2xywh)
from yolocode.yolov9.utils.plots import Annotator, colors, save_one_box
from yolocode.yolov9.utils.torch_utils import select_device, smart_inference_mode


class YOLOv9Thread(QThread):
    # 입출력 메시지
    send_input = Signal(np.ndarray)
    send_output = Signal(np.ndarray)
    send_msg = Signal(str)
    # 상태 표시줄에 데이터 진행 표시줄 데이터 보이기
    send_fps = Signal(str)  # fps
    # send_labels = Signal(dict)  # Detected target results (number of each category)
    send_progress = Signal(int)  # Completeness
    send_class_num = Signal(int)  # Number of categories detected
    send_target_num = Signal(int)  # Targets detected
    send_result_picture = Signal(dict)  # Send the result picture
    send_result_table = Signal(list)    # Send the result table


    def __init__(self):
        super(YOLOv9Thread, self).__init__()
        # YOLOSHOW 인터페이스 매개 변수 설정
        self.current_model_name = None  # The detection model name to use
        self.new_model_name = None  # Models that change in real time
        self.source = None  # input source
        self.stop_dtc = True  # 검사 중지
        self.is_continue = True  # continue/pause
        self.save_res = False  # Save test results
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.labels_dict = {}  # return a dictionary of results
        self.progress_value = 0  # progress bar
        self.res_status = False  # result status
        self.parent_workpath = None  # parent work path

        # YOLOv9 매개변수 설정
        self.model = None
        self.data = 'yolocode/yolov9/data/coco.yaml'  # data_dict
        self.imgsz = (640, 640)
        self.device = ''
        self.dataset = None
        self.task = 'detect'
        self.dnn = False
        self.half = False
        self.agnostic_nms = False
        self.stream_buffer = False
        self.crop_fraction = 1.0
        self.done_warmup = False
        self.vid_path, self.vid_writerm, self.vid_cap = None, None, None
        self.batch = None
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.vid_stride = 1  # 비디오 프레임 속도
        self.max_det = 1000  # 최대 검출 수
        self.classes = None  # 탐지 범주 지정  --class 0, or --class 0 2 3
        self.line_thickness = 3
        self.results_picture = dict()     # 결과 사진
        self.results_table = list()         # 결과표

    def run(self):

        source = str(self.source)
        # 입력 소스 유형 결정
        if isinstance(IMG_FORMATS, str) or isinstance(IMG_FORMATS, tuple):
            self.is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        else:
            self.is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        self.is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam = source.isnumeric() or source.endswith(".streams") or (self.is_url and not self.is_file)
        self.screenshot = source.lower().startswith("screen")
        # 입력 소스가 폴더인지 확인하고, 목록이면 폴더인지 확인.
        self.is_folder = isinstance(self.source, list)
        if self.save_res:
            self.save_path = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
            self.save_path.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(self.device)
        weights = self.new_model_name
        self.current_model_name = self.new_model_name
        self.send_msg.emit("Loading model: {}".format(os.path.basename(self.new_model_name)))
        model = DetectMultiBackend_YOLOv9(weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        vid_stride = self.vid_stride
        dataset_list = []
        if self.webcam:
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif self.screenshot:
            dataset = LoadScreenshots(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        elif self.is_folder:
            for source_i in self.source:
                dataset_list.append(
                    LoadImages(source_i, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride))
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs  # 视频路径 视频写入器
        model.warmup(imgsz=(1 if self.pt or model.triton else bs, 3, *self.imgsz))  # warmup
        self.model = model
        if self.is_folder:
            for dataset in dataset_list:
                self.detect(dataset, device, bs)
        else:
            self.detect(dataset, device, bs)

    def detect(self, dataset, device, bs):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # seen 은 이미지 수를 의미
        datasets = iter(dataset)
        count = 0  # run location frame
        start_time = time.time()  # used to calculate the frame rate
        while True:
            if self.stop_dtc:
                self.send_msg.emit('Stop Detection')
                # --- 이미지 및 테이블 결과 보내기 --- #
                self.send_result_picture.emit(self.results_picture)  # 发送图片结果
                for key, value in self.results_picture.items():
                    self.results_table.append([key, str(value)])
                self.results_picture = dict()
                self.send_result_table.emit(self.results_table)  # 发送表格结果
                self.results_table = list()
                # --- 이미지 및 테이블 결과 보내기 --- #
                # 리소스 해제
                if hasattr(dataset, 'threads'):
                    for thread in dataset.threads:
                        if thread.is_alive():
                            thread.join(timeout=1)  # Add timeout
                if hasattr(dataset, 'cap'):
                    dataset.cap.release()
                cv2.destroyAllWindows()
                if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                    self.vid_writer[-1].release()
                break
            #  모델 교체 여부 판단
            if self.current_model_name != self.new_model_name:
                weights = self.current_model_name
                data = self.data
                self.send_msg.emit(f'Loading Model: {os.path.basename(weights)}')
                self.model = DetectMultiBackend_YOLOv9(weights, device=device, dnn=False, data=data, fp16=False)
                stride, names, pt = self.model.stride, self.model.names, self.model.pt
                imgsz = check_img_size(self.imgsz, s=stride)  # check image size
                self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                self.current_model_name = self.new_model_name
            # 추론 시작
            if self.is_continue:
                if self.is_file:
                    self.send_msg.emit("Detecting File: {}".format(os.path.basename(self.source)))
                elif self.webcam and not self.is_url:
                    self.send_msg.emit("Detecting Webcam: Camera_{}".format(self.source))
                elif self.is_folder:
                    self.send_msg.emit("Detecting Folder: {}".format(os.path.dirname(self.source[0])))
                elif self.is_url:
                    self.send_msg.emit("Detecting URL: {}".format(self.source))
                else:
                    self.send_msg.emit("Detecting: {}".format(self.source))
                path, im, im0s, self.vid_cap, s = next(datasets)
                # 원본 이미지가 입력 상자로 전송됩니다.
                self.send_input.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                count += 1
                percent = 0  # 진행률 표시줄
                # processBar 처리
                if self.vid_cap:
                    percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.progress_value)
                    self.send_progress.emit(percent)
                else:
                    percent = self.progress_value
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.send_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                with dt[0]:
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = False
                    pred = self.model(im, augment=False, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                               max_det=self.max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if self.webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    if self.save_res:
                        save_path = str(self.save_path / p.name)  # im.jpg
                        self.res_path = save_path
                    annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                    # 카테고리 타겟 수
                    class_nums = 0
                    target_nums = 0
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            class_nums += 1
                            target_nums += int(n)
                            if self.names[int(c)] in self.labels_dict:
                                self.labels_dict[self.names[int(c)]] += int(n)
                            else:  # 카테고리의 첫 번째 발생
                                self.labels_dict[self.names[int(c)]] = int(n)

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{self.names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # Stream results
                    im0 = annotator.result()
                    # 결과 보내기
                    im0 = annotator.result()
                    self.send_output.emit(im0)  # 이미지 출력
                    self.send_class_num.emit(class_nums)
                    self.send_target_num.emit(target_nums)
                    self.results_picture = self.labels_dict

                    # Save results (image with detections)
                    if self.save_res:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path[i] != save_path:  # new video
                                self.vid_path[i] = save_path
                                if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                    self.vid_writer[i].release()  # release previous video writer
                                if self.vid_cap:  # video
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(
                                    Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                     (w, h))
                            self.vid_writer[i].write(im0)

                    if self.speed_thres != 0:
                        time.sleep(self.speed_thres / 1000)  # delay , ms

                if percent == self.progress_value and not self.webcam:
                    self.send_progress.emit(0)
                    self.send_msg.emit('Finish Detection')
                    # --- 이미지 및 표 결과 보내기 --- #
                    self.send_result_picture.emit(self.results_picture)  # 이미지 결과 보내기
                    for key, value in self.results_picture.items():
                        self.results_table.append([key, str(value)])
                    self.results_picture = dict()
                    self.send_result_table.emit(self.results_table)  # 결과 내보내기
                    self.results_table = list()
                    # --- 이미지 및 표 결과 보내기 --- #
                    self.res_status = True
                    if self.vid_cap is not None:
                        self.vid_cap.release()
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    break
