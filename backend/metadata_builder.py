import os
import pickle
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from config import SystemConfig
from utils.custom_logging import custom_logger
import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class MetadataBuilder:
    """
    Builds and manages metadata for images and faces.
    """
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.metadata_file = os.path.join(image_dir, 'general_metadata.pkl')
        self.metadata = self.load_metadata()

    def load_metadata(self) -> Dict:
        """Load metadata from file if it exists."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def build_metadata(self):
        """Build metadata for all images in the directory."""
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    if image_path not in self.metadata:
                        try:
                            with Image.open(image_path) as img:
                                ctime = os.path.getctime(image_path)
                                readable_date = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
                                self.metadata[image_path] = {
                                    'size': img.size,
                                    'mode': img.mode,
                                    'format': img.format,
                                    'date': readable_date,
                                    'camera': img.getexif().get(271) if hasattr(img, 'getexif') and img.getexif() else None,
                                    'model': img.getexif().get(272) if hasattr(img, 'getexif') and img.getexif() else None,
                                }
                                custom_logger.info(f"Processed metadata for {image_path}")
                            
                        except Exception as e:
                            custom_logger.error(f"Error processing {image_path}: {e}")
                            
        self.save_metadata()
        custom_logger.info("Metadata building complete.")
        return self.metadata

if __name__ == "__main__":
    IMAGE_DIR = "<<YourSampleImageDir>>"    
    builder = MetadataBuilder(image_dir=IMAGE_DIR)
    metadata = builder.build_metadata()
    print(f"Built metadata for {len(metadata)} images.")
    print(metadata)