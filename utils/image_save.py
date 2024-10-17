from PIL import Image
import pillow_heif
import cv2
import os

class ImageSaver:
    def __init__(self, plotted_img):
        """
        Initializes the ImageSaver with the plotted image and dataset mode.
        Registers HEIF/HEIC opener globally.
        """
        self.plotted_img = plotted_img

        # Register HEIF/HEIC support globally
        pillow_heif.register_heif_opener()

    def save_image(self, save_path):
        """
        Save video predictions as mp4 at specified path or images as HEIC/HEIF or other formats.
        """
        im0 = self.plotted_img  # This is a 3-channel ndarray (BGR format)

        root, ext = os.path.splitext(save_path)
        if ext.lower() == '.heic' or ext.lower() == '.heif':
            # Convert the image from BGR to RGB (since HEIC expects RGB)
            im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            im0_pil = Image.fromarray(im0_rgb)
            # Save as HEIC/HEIF
            im0_pil.save(save_path, format='HEIF')  # Use 'HEIF' format for saving
        else:
            # Save using OpenCV for other formats (e.g., PNG, JPG)
            cv2.imwrite(save_path, im0)
