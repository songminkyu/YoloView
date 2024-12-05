import hashlib

class DatasetHashClac:
    def __init__(self):
        pass

    @staticmethod
    def get_file_hash(file_path):
        """Calculate the SHA1 hash of a file."""
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
