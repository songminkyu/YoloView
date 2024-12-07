import os
import shutil
from DatasetHashClac import DatasetHashClac

class DatasetCleaner:
    def __init__(self, root_folder, subfolders, is_delete, classification_folder):
        self.root_folder = root_folder
        self.subfolders = subfolders
        self.is_delete = is_delete
        self.classification_folder = classification_folder

    def process_zero_size_data(self, images_folder, labels_folder):
        for label_file_path in os.listdir(labels_folder):
            label_file_path = os.path.join(labels_folder, label_file_path)
            if os.path.getsize(label_file_path) == 0:
                base_name = os.path.splitext(os.path.basename(label_file_path))[0]
                image_files = [f for f in os.listdir(images_folder) if f.startswith(base_name)]
                if self.is_delete:
                    os.remove(label_file_path)
                    print(f"Deleted label file: {label_file_path}")
                    for image_file in image_files:
                        os.remove(os.path.join(images_folder, image_file))
                        print(f"Deleted image file: {os.path.join(images_folder, image_file)}")
                else:
                    dest_images = os.path.join(self.classification_folder, "images")
                    dest_labels = os.path.join(self.classification_folder, "labels")
                    os.makedirs(dest_images, exist_ok=True)
                    os.makedirs(dest_labels, exist_ok=True)
                    shutil.move(label_file_path, os.path.join(dest_labels, os.path.basename(label_file_path)))
                    for image_file in image_files:
                        shutil.move(os.path.join(images_folder, image_file), os.path.join(dest_images, image_file))

    def remove_duplicate_images_and_labels(self, images_folder, labels_folder):
        hash_calculator = DatasetHashClac()
        hash_dict = {}
        duplicate_images = []
        duplicate_labels = []

        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            try:
                file_hash = hash_calculator.get_file_hash(image_path)
                if file_hash in hash_dict:
                    duplicate_images.append(image_path)
                    original_path = hash_dict[file_hash]
                    print(f"Duplicate image found: {image_path} (Original: {original_path})")

                    base_name = os.path.splitext(image_file)[0]
                    matching_labels = [f for f in os.listdir(labels_folder) if f.startswith(base_name)]
                    duplicate_labels.extend([os.path.join(labels_folder, f) for f in matching_labels])
                else:
                    hash_dict[file_hash] = image_path
            except Exception as e:
                print(f"Error calculating hash for {image_path}: {e}")

        for duplicate in duplicate_images:
            os.remove(duplicate)
            print(f"Deleted duplicate image: {duplicate}")

        for duplicate in duplicate_labels:
            os.remove(duplicate)
            print(f"Deleted duplicate label: {duplicate}")

    def remove_unmatched_images_and_labels_proc(self, images_folder, labels_folder):
        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            base_name = os.path.splitext(label_file)[0]
            image_path = os.path.join(images_folder, f"{base_name}.jpg")
            if not os.path.exists(image_path):
                os.remove(label_path)
                print(f"Deleted unmatched label: {label_path}")

    def remove_zero_and_duplicate(self):
        for subfolder in self.subfolders:
            subfolder_path = os.path.join(self.root_folder, subfolder)
            if os.path.exists(subfolder_path):
                images_folder = os.path.join(subfolder_path, "images")
                labels_folder = os.path.join(subfolder_path, "labels")
                if os.path.exists(images_folder) and os.path.exists(labels_folder):
                    self.process_zero_size_data(images_folder, labels_folder)
                    self.remove_duplicate_images_and_labels(images_folder, labels_folder)
                else:
                    print(f"Missing images or labels folder in: {subfolder_path}")
            else:
                print(f"Folder does not exist: {subfolder_path}")

    def remove_segments(self):
        for subfolder in self.subfolders:
            labels_folder = os.path.join(self.root_folder, subfolder, "labels")
            if os.path.exists(labels_folder):
                for label_file in os.listdir(labels_folder):
                    label_path = os.path.join(labels_folder, label_file)
                    with open(label_path, "r") as file:
                        lines = file.readlines()

                    if any(len(line.split()) > 5 for line in lines):
                        os.remove(label_path)
                        print(f"Deleted segment label file: {label_path}")

    def remove_bounding_boxes(self):
        for subfolder in self.subfolders:
            labels_folder = os.path.join(self.root_folder, subfolder, "labels")
            if os.path.exists(labels_folder):
                for label_file in os.listdir(labels_folder):
                    label_path = os.path.join(labels_folder, label_file)
                    with open(label_path, "r") as file:
                        lines = file.readlines()

                    if all(len(line.split()) == 5 for line in lines):
                        os.remove(label_path)
                        print(f"Deleted bounding box label file: {label_path}")

    def remove_labels_and_images_by_class_ids(self, class_ids_to_remove):
        for subfolder in self.subfolders:
            labels_folder = os.path.join(self.root_folder, subfolder, "labels")
            images_folder = os.path.join(self.root_folder, subfolder, "images")
            if os.path.exists(labels_folder):
                for label_file in os.listdir(labels_folder):
                    label_path = os.path.join(labels_folder, label_file)
                    with open(label_path, "r") as file:
                        lines = file.readlines()

                    should_delete = False
                    for line in lines:
                        class_id = int(line.split()[0])
                        if class_id in class_ids_to_remove:
                            should_delete = True
                            break

                    if should_delete:
                        os.remove(label_path)
                        print(f"Deleted label file: {label_path}")
                        image_path = os.path.join(images_folder, f"{os.path.splitext(label_file)[0]}.jpg")
                        if os.path.exists(image_path):
                            os.remove(image_path)
                            print(f"Deleted image file: {image_path}")

    def remove_unmatched_images_and_labels(self):
        print("Removing unmatched labels and images...")
        for subfolder in self.subfolders:
            subfolder_path = os.path.join(self.root_folder, subfolder)

            if os.path.exists(subfolder_path):
                images_folder = os.path.join(subfolder_path, "images")
                labels_folder = os.path.join(subfolder_path, "labels")

                if os.path.exists(images_folder) and os.path.exists(labels_folder):
                    self.remove_unmatched_images_and_labels_proc(images_folder, labels_folder)

