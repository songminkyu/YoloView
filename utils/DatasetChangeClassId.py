import os

class DatasetChangeClassId:
    def __init__(self, root_folder, subfolders, target_class_ids, new_class_id):
        self.root_folder = root_folder
        self.subfolders = subfolders
        self.target_class_ids = target_class_ids
        self.new_class_id = new_class_id

    def update_class_id_processing(self):
        for subfolder in self.subfolders:
            labels_folder = os.path.join(self.root_folder, subfolder, "labels")
            if os.path.exists(labels_folder):
                for label_file_path in os.listdir(labels_folder):
                    label_file_path = os.path.join(labels_folder, label_file_path)
                    if label_file_path.endswith(".txt"):
                        with open(label_file_path, "r") as file:
                            lines = file.readlines()

                        has_changes = False
                        for i in range(len(lines)):
                            elements = lines[i].strip().split()
                            if elements and int(elements[0]) in self.target_class_ids:
                                elements[0] = str(self.new_class_id)
                                lines[i] = " ".join(elements) + "\n"
                                has_changes = True

                        if has_changes:
                            with open(label_file_path, "w") as file:
                                file.writelines(lines)
                            print(f"Class ID updated: {label_file_path}")
