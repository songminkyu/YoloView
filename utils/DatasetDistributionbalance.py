import os
import random
import shutil
import yaml

class DatasetDistributionbalance:
    def __init__(self, root_folder, train_ratio=0.7, test_ratio=0.15, valid_ratio=0.15):
        self.root_folder = root_folder
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio

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

            new_valid_set.extend(class_data[train_count:train_count + valid_count])
            new_test_set.extend(class_data[train_count + valid_count:])

        self.move_dataset(new_valid_set, os.path.join(self.root_folder, "valid"))
        self.move_dataset(new_test_set, os.path.join(self.root_folder, "test"))

    def match_images_and_labels(self, images_folder, labels_folder):
        image_files = {os.path.splitext(f)[0]: os.path.join(images_folder, f) for f in os.listdir(images_folder)}
        label_files = {os.path.splitext(f)[0]: os.path.join(labels_folder, f) for f in os.listdir(labels_folder)}

        return [(image_files[k], label_files[k]) for k in image_files.keys() & label_files.keys()]

    def group_files_by_class(self, matched_files, class_names):
        data_by_class = {name: [] for name in class_names}
        for image_path, label_path in matched_files:
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    if class_id < len(class_names):
                        data_by_class[class_names[class_id]].append((image_path, label_path))
        return data_by_class

    def move_dataset(self, dataset, target_folder):
        images_folder = os.path.join(target_folder, "images")
        labels_folder = os.path.join(target_folder, "labels")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        for image_path, label_path in dataset:
            shutil.move(image_path, os.path.join(images_folder, os.path.basename(image_path)))
            shutil.move(label_path, os.path.join(labels_folder, os.path.basename(label_path)))
