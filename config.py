import torch
class Config:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
        self.device  = "cuda" if torch.cuda.is_available() else "cpu" ## mps causes issues with faiss
        self.image_to_image_model : str = "google/siglip-base-patch16-384"
        self.my_img_directory : str = "/Users/varamana/Desktop/Wiki"
        self.sample_query_img_path : str = "/Users/varamana/Desktop/Wiki/20211221_123756.jpg"

SystemConfig = Config()