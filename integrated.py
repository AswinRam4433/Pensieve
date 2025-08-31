import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from PIL import Image
import os
import requests
from PIL import Image
from io import BytesIO

from face_recog import FaceIndexer
from clip_search import CLIPImageSearcher
from pipeline import ImageToImageSearch

class IntegratedPipeline:
    def __init__(self, image_dir:str) -> None:
        self.image_dir = image_dir
        self.FaceIndexer = FaceIndexer(image_dir=image_dir)
        self.CLIPImageSearcher = CLIPImageSearcher(image_dir=image_dir)
        self.ImageToImageSearch = ImageToImageSearch(image_dir=image_dir)
    
    def scan_repo(self):
        self.FaceIndexer.build_index()
        self.CLIPImageSearcher.build_index()
        self.ImageToImageSearch.build_index()
    
    def get_index_info(self):
        ## TODO: Make all the names uniform and consistent
        self.FaceIndexer.load_index() ## TODO: Need to just get index info instead of actual loading here
        self.CLIPImageSearcher.get_index_info()
        self.ImageToImageSearch.get_index_info()
    
        