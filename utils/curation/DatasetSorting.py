import os
from .DatasetHashClac import DatasetHashClac

class DatasetSorting:
    def __init__(self, root_folder, subfolders):
        self.root_folder = root_folder
        self.subfolders = subfolders

    def sort_files_to_match_processing(self):
        for subfolder in self.subfolders:
            subfolder_path = os.path.join(self.root_folder, subfolder)
            images_folder = os.path.join(subfolder_path, "images")
            labels_folder = os.path.join(subfolder_path, "labels")
            if os.path.exists(images_folder) and os.path.exists(labels_folder):
                self.sort_files_to_match(images_folder, labels_folder)

    def sort_files_to_match(self, images_folder, labels_folder):
        hash_calculator = DatasetHashClac()
        image_files = os.listdir(images_folder)
        for index, image_file in enumerate(image_files, start=1):
            base_name = os.path.splitext(image_file)[0]
            label_files = [f for f in os.listdir(labels_folder) if f.startswith(base_name)]
            if label_files:
                image_path = os.path.join(images_folder, image_file)
                label_path = os.path.join(labels_folder, label_files[0])

                sha1_hash = hash_calculator.get_file_hash(image_path)
                new_base_name = f"{index:05d}_{sha1_hash}"
                new_image_path = os.path.join(images_folder, f"{new_base_name}{os.path.splitext(image_file)[1]}")
                new_label_path = os.path.join(labels_folder, f"{new_base_name}{os.path.splitext(label_files[0])[1]}")

                os.rename(image_path, new_image_path)
                os.rename(label_path, new_label_path)
                print(f"Renamed: {image_file} -> {new_image_path}")
