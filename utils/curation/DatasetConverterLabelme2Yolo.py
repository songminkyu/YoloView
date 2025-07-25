"""
Updated on Feb 08, 2024 to make it compatible with Yolov8
@author: spatiallysaying(Durga Prasad,D)

"""
import base64
import glob
import io
import json
import math
import os
import random
import shutil
import uuid
import logging

import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import cv2
import numpy as np
import tqdm
import argparse

# set seed
random.seed(12345678)
random.Random().seed(12345678)
np.random.seed(12345678)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("labelme2yolov8")


def train_test_split(dataset_index, test_size=0.2):
    """Split dataset into train set and test set with test_size"""
    test_size = min(max(0.0, test_size), 1.0)
    total_size = len(dataset_index)
    train_size = int(round(total_size * (1.0 - test_size)))
    random.shuffle(dataset_index)
    train_index = dataset_index[:train_size]
    test_index = dataset_index[train_size:]

    return train_index, test_index


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_pil(img_data):
    """Convert img_data(byte) to PIL.Image"""
    file = io.BytesIO()
    file.write(img_data)
    img_pil = PIL.Image.open(file)
    return img_pil


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_arr(img_data):
    """Convert img_data(byte) to numpy.ndarray"""
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_b64_to_arr(img_b64):
    """Convert img_b64(str) to numpy.ndarray"""
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_pil_to_data(img_pil):
    """Convert PIL.Image to img_data(byte)"""
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_data = file.getvalue()
    return img_data


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_arr_to_b64(img_arr):
    """Convert numpy.ndarray to img_b64(str)"""
    img_pil = PIL.Image.fromarray(img_arr)
    file = io.BytesIO()
    img_pil.save(file, format="PNG")
    img_bin = file.getvalue()
    img_b64 = base64.encodebytes(img_bin)
    return img_b64


# copy form https://github.com/wkentaro/labelme/blob/main/labelme/utils/image.py
def img_data_to_png_data(img_data):
    """Convert img_data(byte) to png_data(byte)"""
    with io.BytesIO() as f_out:
        f_out.write(img_data)
        img = PIL.Image.open(f_out)

        with io.BytesIO() as f_in:
            img.save(f_in, "PNG")
            f_in.seek(0)
            return f_in.read()


def extend_point_list(point_list, out_format="polygon"):
    """Extend point list to polygon or bbox"""
    x_min = min(float(point) for point in point_list[::2])
    x_max = max(float(point) for point in point_list[::2])
    y_min = min(float(point) for point in point_list[1::2])
    y_max = max(float(point) for point in point_list[1::2])

    if out_format == "bbox":
        x_i = x_min
        y_i = y_min
        w_i = x_max - x_min
        h_i = y_max - y_min
        x_i = x_i + w_i / 2
        y_i = y_i + h_i / 2
        return np.array([x_i, y_i, w_i, h_i])

    return np.array([x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max])


def save_yolo_label(obj_list, label_dir, target_dir, target_name):
    """Save yolo label to txt file"""
    txt_path = os.path.join(label_dir, target_dir, target_name)

    with open(txt_path, "w+", encoding="utf-8") as file:
        for label, points in obj_list:
            points = [str(item) for item in points]
            line = f"{label} {' '.join(points)}\n"
            file.write(line)


def save_yolo_image(json_data, json_dir, image_dir, target_dir, target_name):
    """Save yolo image to image_dir_path/target_dir"""
    img_path = os.path.join(image_dir, target_dir, target_name)

    if json_data["imageData"] is None:
        image_name = json_data["imagePath"]
        src_image_name = os.path.join(json_dir, image_name)
        src_image = cv2.imread(src_image_name)
        cv2.imwrite(img_path, src_image)
    else:
        img = img_b64_to_arr(json_data["imageData"])
        PIL.Image.fromarray(img).save(img_path)

    return img_path


class Labelme2YOLOv8:
    """Labelme to YOLO format converter"""

    def __init__(self, json_dir, output_format, label_list):
        self._json_dir = os.path.expanduser(json_dir)
        self._output_format = output_format
        self._label_list = []
        self._label_id_map = {}
        self._label_dir_path = ""
        self._image_dir_path = ""

        print(label_list)
        if label_list:
            self._label_list = label_list
            self._label_id_map = {
                label: label_id for label_id, label in enumerate(label_list)
            }
        print(self._label_id_map)

    def _update_id_map(self, label: str):
        if label not in self._label_list:
            self._label_list.append(label)
            self._label_id_map[label] = len(self._label_id_map)

    def _make_train_val_dir(self, create_test_dir=False):
        self._dataset_dir_path = os.path.join(self._json_dir, "YOLOv8Dataset/")

        # Define base paths for test, train, and val inside the YOLOv8Dataset directory
        parts = ["train", "valid", "test"] if create_test_dir else ["train", "valid"]
        dirs = [os.path.join(self._dataset_dir_path, part) for part in parts]

        # Cleanup: Remove existing directories to ensure a clean state
        for part_path in dirs:
            shutil.rmtree(part_path, ignore_errors=True)

        # Create new directory structure
        for part in parts:
            image_dir_path = os.path.join(self._dataset_dir_path, part, "images/")
            label_dir_path = os.path.join(self._dataset_dir_path, part, "labels/")
            for dir_path in [image_dir_path, label_dir_path]:
                os.makedirs(dir_path, exist_ok=True)

    def _get_dataset_part_json_names(self, dataset_part: str):
        """Get json names in dataset_part folder"""
        set_folder = os.path.join(self._json_dir, dataset_part)
        json_names = []
        for sample_name in os.listdir(set_folder):
            set_dir = os.path.join(set_folder, sample_name)
            if os.path.isdir(set_dir):
                json_names.append(sample_name + ".json")
        return json_names

    def _train_test_split(self, json_names, val_size, test_size=None):
        """Split json names to train, val, test"""
        total_size = len(json_names)
        dataset_index = list(range(total_size))
        train_ids, val_ids = train_test_split(dataset_index, test_size=val_size)
        test_ids = []
        if test_size is None:
            test_size = 0.0
        if test_size > 0.0:
            train_ids, test_ids = train_test_split(
                train_ids, test_size=test_size / (1 - val_size)
            )
        train_json_names = [json_names[train_idx] for train_idx in train_ids]
        val_json_names = [json_names[val_idx] for val_idx in val_ids]
        test_json_names = [json_names[test_idx] for test_idx in test_ids]

        return train_json_names, val_json_names, test_json_names

    def convert(self, val_size, test_size):
        """Convert labelme format to yolo format"""
        json_names = glob.glob(
            os.path.join(self._json_dir, "**", "*.json"), recursive=True
        )
        json_names = sorted(json_names)

        train_json_names, val_json_names, test_json_names = self._train_test_split(
            json_names, val_size, test_size
        )

        self._make_train_val_dir(test_size > 0.0)

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        dirs = ("train", "valid", "test")
        names = (train_json_names, val_json_names, test_json_names)
        for target_dir, json_names in zip(dirs, names):
            logger.info("Converting %s set ...", target_dir)
            for json_name in tqdm.tqdm(json_names):
                self.covert_json_to_text(target_dir, json_name)

        self._save_dataset_yaml()

    def covert_json_to_text(self, target_dir, json_name):
        """Convert one json file to yolo format text file and save them to files"""
        with open(json_name, encoding="utf-8") as file:
            json_data = json.load(file)

        # filename: str = uuid.UUID(int=random.Random().getrandbits(128)).hex

        # Extract only file names excluding path and extension
        filename = os.path.splitext(os.path.basename(json_name))[0]
        image_name = f"{filename}.png"
        label_name = f"{filename}.txt"

        # Correctly updated paths for saving images and labels according to the new directory structure
        img_path = save_yolo_image(
            json_data,
            self._json_dir,
            self._dataset_dir_path,
            os.path.join(target_dir, "images"),
            image_name  # Passing image_name directly, assuming the image_dir includes the target directory
        )
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        # txt_path = os.path.join(label_dir, target_dir, target_name)
        save_yolo_label(
            yolo_obj_list,
            self._dataset_dir_path,
            os.path.join(target_dir, "labels"),
            label_name  # Passing label_name directly, assuming the label_dir includes the target directory
        )

    def convert_one(self, json_name):
        """Convert one json file to yolo format text file and save them to files"""
        json_path = os.path.join(self._json_dir, json_name)
        with open(json_path, encoding="utf-8") as file:
            json_data = json.load(file)

        # Determine where to save the converted files (train, val, or test)
        # This example assumes a default directory, adjust as needed
        target_dir = "train"  # or "valid" or "test", depending on your logic

        image_name = json_name.replace(".json", ".png")
        label_name = json_name.replace(".json", ".txt")

        # Update paths for saving images and labels
        img_path = save_yolo_image(
            json_data,
            self._json_dir,
            self._dataset_dir_path,
            os.path.join(target_dir, "images"),
            image_name
        )
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        save_yolo_label(
            yolo_obj_list,
            self._dataset_dir_path,
            os.path.join(target_dir, "labels"),
            label_name
        )

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data["shapes"]:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape["shape_type"] == "circle":
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)

            if yolo_obj:
                yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape["points"][0]

        radius = math.sqrt(
            (obj_center_x - shape["points"][1][0]) ** 2
            + (obj_center_y - shape["points"][1][1]) ** 2
        )
        num_points = 36
        points = np.zeros(2 * num_points)
        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points
            points[2 * i] = (obj_center_x + radius * math.cos(angle)) / img_w
            points[2 * i + 1] = (obj_center_y + radius * math.sin(angle)) / img_h

        if shape["label"]:
            label = shape["label"]
            if label not in self._label_list:
                self._update_id_map(label)
            label_id = self._label_id_map[shape["label"]]

            return label_id, points.tolist()

        return None

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        point_list = shape["points"]
        points = np.zeros(2 * len(point_list))
        points[::2] = [float(point[0]) / img_w for point in point_list]
        points[1::2] = [float(point[1]) / img_h for point in point_list]

        if len(points) == 4:
            if self._output_format == "polygon":
                points = extend_point_list(points)
            if self._output_format == "bbox":
                points = extend_point_list(points, "bbox")

        if shape["label"]:
            label = shape["label"]
            if label not in self._label_list:
                self._update_id_map(label)
            label_id = self._label_id_map[shape["label"]]

            return label_id, points.tolist()

        return None

    # data.yaml 생성 및 학습데이터 메타 정보 기록
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, "YOLOv8Dataset/", "data.yaml")

        with open(yaml_path, "w+", encoding="utf-8") as yaml_file:

            # Make sure to reference images subdirectories
            train_images_dir = '../train/images'
            val_images_dir = '../valid/images'
            test_images_dir = '../test/images'

            names_str = ""
            for label, _ in self._label_id_map.items():
                names_str += f'"{label}", '
            names_str = names_str.rstrip(", ")

            if os.path.exists(test_images_dir):
                content = (
                    f"train: {train_images_dir}\nval: {val_images_dir}\ntest: {test_images_dir}\n"
                    f"nc: {len(self._label_id_map)}\n"
                    f"names: [{names_str}]"
                )
            else:
                content = (
                    f"train: {train_images_dir}\nval: {val_images_dir}\n"
                    f"nc: {len(self._label_id_map)}\n"
                    f"names: [{names_str}]"
                )

            yaml_file.write(content)


def main():
    parser = argparse.ArgumentParser("labelme2yolov8")

    parser.add_argument(
        "--json_dir", type=str, help="Please input the path of the labelme json files."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=0.2,
        help="Please input the validation dataset size, for example 0.2.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=0.0,
        help="Please input the test dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        nargs="?",
        default=None,
        help="If you put json name, it would convert only one json file to YOLO.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="polygon",
        help='The default output format for labelme2yolov8 is "polygon".'
             ' However, you can choose to output in bbox format by specifying the "bbox" option.',
    )
    parser.add_argument(
        "--label_list",
        type=str,
        nargs="+",
        default=None,
        help="The ordered label list, for example --label_list cat dog",
        required=False,
    )

    args = parser.parse_args()

    args.json_dir = "d:\\00_OpenCV_Project\\Study_at_Australia\\python\\yoloshow\\re"
    args.output_format = "bbox"  # bbox or polygon

    if not args.json_dir:
        parser.print_help()
        return 0

    convertor = Labelme2YOLOv8(args.json_dir, args.output_format, args.label_list)

    if args.json_name is None:
        convertor.convert(val_size=args.val_size, test_size=args.test_size)
    else:
        convertor.convert_one(args.json_name)

    return 0

if __name__ == "__main__":
    main()

