import os
import random
import shutil
import yaml
import glob
from concurrent.futures import ThreadPoolExecutor

class DatasetDistributionbalance:
    def __init__(self, root_folder, train_ratio=0.7, test_ratio=0.15, valid_ratio=0.15, num_workers=8):
        self.root_folder = root_folder
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.num_workers = num_workers  # 병렬 처리를 위한 워커 스레드 수

    def adjust_dataset_splits(self):
        yaml_path = os.path.join(self.root_folder, "data.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError("YAML file not found in the root folder.")

        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        if "nc" not in yaml_data or "names" not in yaml_data:
            raise ValueError("YAML file must contain 'nc' and 'names' fields.")

        class_names = yaml_data["names"]

        images_folder = os.path.join(self.root_folder, "train", "images")
        labels_folder = os.path.join(self.root_folder, "train", "labels")

        if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
            raise FileNotFoundError("Train folder not found in the root directory.")

        matched_files = self.match_images_and_labels(images_folder, labels_folder)
        data_by_class = self.group_files_by_class(matched_files, class_names)

        new_valid_set = []
        new_test_set = []

        for class_data in data_by_class.values():
            random.shuffle(class_data)
            total_count = len(class_data)
            train_count = int(total_count * self.train_ratio)
            valid_count = int(total_count * self.valid_ratio)
            # 나머지는 test_count가 됨
            # train : [0 : train_count]
            # valid : [train_count : train_count + valid_count]
            # test  : [train_count + valid_count : ]

            new_valid_set.extend(class_data[train_count:train_count + valid_count])
            new_test_set.extend(class_data[train_count + valid_count:])

        self.move_dataset(new_valid_set, os.path.join(self.root_folder, "valid"))
        self.move_dataset(new_test_set, os.path.join(self.root_folder, "test"))

    def match_images_and_labels(self, images_folder, labels_folder):
        # glob을 사용해 이미지와 라벨 파일을 찾는다. 다양한 확장자 고려 시 "*.*"
        image_files = {os.path.splitext(os.path.basename(f))[0]: f for f in
                       glob.glob(os.path.join(images_folder, "*.*"))}
        label_files = {os.path.splitext(os.path.basename(f))[0]: f for f in
                       glob.glob(os.path.join(labels_folder, "*.*"))}

        keys = image_files.keys() & label_files.keys()
        return [(image_files[k], label_files[k]) for k in keys]

    def group_files_by_class(self, matched_files, class_names):
        data_by_class = {name: [] for name in class_names}
        for image_path, label_path in matched_files:
            # 메모리 사용 최소화를 위해 readlines() 대신 한줄씩 순회
            with open(label_path, 'r') as label_file:
                for line in label_file:
                    if not line.strip():
                        continue
                    class_id = int(line.split()[0])
                    if class_id < len(class_names):
                        data_by_class[class_names[class_id]].append((image_path, label_path))
        return data_by_class

    def move_dataset(self, dataset, target_folder):
        images_folder = os.path.join(target_folder, "images")
        labels_folder = os.path.join(target_folder, "labels")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        # 파일 이동을 병렬 처리
        def move_files(image_label_pair):
            image_path, label_path = image_label_pair
            shutil.move(image_path, os.path.join(images_folder, os.path.basename(image_path)))
            shutil.move(label_path, os.path.join(labels_folder, os.path.basename(label_path)))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            executor.map(move_files, dataset)
