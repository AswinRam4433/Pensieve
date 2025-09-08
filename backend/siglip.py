## NOT USED ANYMORE
## FOR INITIAL TESTING PURPOSES ONLY
import torch
import faiss
from torchvision import transforms

from PIL import Image
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel, AutoTokenizer

import numpy as np
import requests

device = torch.device('cuda' if torch.cuda.is_available() else "cpu") ## using mps here causes segmentation faults

model = SiglipModel.from_pretrained("google/siglip-base-patch16-384").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-384")

def add_vector(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)

def embed_siglip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features
    
API_TOKEN="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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/rows?dataset=ceyda/fashion-products-small&config=default&split=train"

def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()

index = faiss.IndexFlatL2(768)

# read the image and add vector
for elem in data["rows"]:
  url = elem["row"]["image"]["src"]
  image = Image.open(requests.get(url, stream=True).raw)
  #Generate Embedding of Image
  clip_features = embed_siglip(image)
  #Add vector to FAISS
  add_vector(clip_features,index)

#Save the index 
faiss.write_index(index,"./siglip_70k.index")


url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsZ4PhHTilpQ5zsG51SPZVrgEhdSfQ7_cg1g&s"
image = Image.open(requests.get(url, stream=True).raw)

with torch.no_grad():
  inputs = processor(images=image, return_tensors="pt").to(device)
  input_features = model.get_image_features(**inputs)

input_features = input_features.detach().cpu().numpy()
input_features = np.float32(input_features)
faiss.normalize_L2(input_features)
distances, indices = index.search(input_features, 3)

for elem in indices[0]:
  url = data["rows"][elem]["row"]["image"]["src"]
  image = Image.open(requests.get(url, stream=True).raw)
  width = 300
  ratio = (width / float(image.size[0]))
  height = int((float(image.size[1]) * float(ratio)))
  img = image.resize((width, height), Image.Resampling.LANCZOS)
  img.show()