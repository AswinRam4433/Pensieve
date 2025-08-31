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

class ImageToImageSearch:
    
    def __init__(self,image_dir:str, model_path="google/siglip-base-patch16-384", tokenizer_path="google/siglip-base-patch16-384", processor_path="google/siglip-base-patch16-384", index_path="./siglip_index.faiss", metadata_path="./index_metadata.json"):
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.image_dir = image_dir
        # File paths for persistence
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        ## image embedding processing pipeline
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.processor = AutoProcessor.from_pretrained(processor_path)

        # Define the FAISS index
        self.INDEX_DIM = self.model.config.text_config.hidden_size
        self.index = faiss.IndexFlatL2(self.INDEX_DIM)
        
        # Metadata to track image URLs and other info
        self.image_metadata = []
        
        # Try to load existing index
        self.load_index()

    def build_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata (image URLs and other info)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.image_metadata, f, indent=2)
            
            print(f"Index saved to {self.index_path}")
            print(f"Metadata saved to {self.metadata_path}")
            print(f"Total vectors in index: {self.index.ntotal}")
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load existing FAISS index and metadata from disk"""
        try:
            if Path(self.index_path).exists() and Path(self.metadata_path).exists():
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.image_metadata = json.load(f)
                
                print(f"Index loaded from {self.index_path}")
                print(f"Metadata loaded from {self.metadata_path}")
                print(f"Total vectors in index: {self.index.ntotal}")
                print(f"Total metadata entries: {len(self.image_metadata)}")
            else:
                print("No existing index found. Starting with empty index.")
        except Exception as e:
            print(f"Error loading index: {e}")
            print("Starting with empty index.")
            self.index = faiss.IndexFlatL2(self.INDEX_DIM)
            self.image_metadata = []
    
    def clear_index(self):
        """Clear the entire index and metadata"""
        try:
            self.index.reset() 
            self.image_metadata = []
            print("Index and metadata cleared successfully.")
            print(f"Index now contains {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error clearing index: {e}")
    
    def get_index_info(self):
        """Get information about the current index"""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.INDEX_DIM,
            "metadata_entries": len(self.image_metadata),
            "index_path": self.index_path,
            "metadata_path": self.metadata_path
        }

    def embed_img(self, image) -> torch.Tensor:
        try:
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                return image_features
        except Exception as e:
            print("Error in embedding image: ", e)
            return torch.Tensor()
    
    def add_vector(self, embedding: torch.Tensor, metadata: dict = None):
        """Add a vector to the index with optional metadata"""
        try:
            vector = embedding.detach().cpu().numpy()
            vector = np.float32(vector)  # using float64 causes segmentation fault downstream
            faiss.normalize_L2(vector)
            
            self.index.add(vector)
            
            if metadata is None:
                metadata = {}
            
            metadata['index_id'] = self.index.ntotal - 1
            self.image_metadata.append(metadata)
            
        except Exception as e:
            print("Error in adding vector to index: ", e)
            return
    
    def img_search(self, image, top_k=3):
        try:
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                input_features = self.model.get_image_features(**inputs)

            input_features = input_features.detach().cpu().numpy()
            input_features = np.float32(input_features)
            faiss.normalize_L2(input_features)
            distances, indices = self.index.search(input_features, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                single_result = {
                    'distance': float(distance),
                    'index': int(idx),
                    'metadata': self.image_metadata[idx] if idx < len(self.image_metadata) else {}
                }
                results.append(single_result)
            
            return results
        except Exception as e:
            print("Error in searching image: ", e)
            return []
    
    def add_images_from_urls(self, image_urls, batch_size=10):
        """Add multiple images from URLs to the index"""
        import requests
        from PIL import Image
        from io import BytesIO
        
        successfully_added = 0
        
        for i, url in enumerate(image_urls):
            try:
                print(f"Processing image {i+1}/{len(image_urls)}: {url}")
                
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content)).convert("RGB")
                embedding = self.embed_img(image)
                
                # Add metadata
                metadata = {
                    'url': url,
                    'added_at': str(pd.Timestamp.now()) if 'pd' in globals() else str(i),
                    'original_index': i
                }
                
                self.add_vector(embedding, metadata)
                successfully_added += 1
                
                # Save periodically
                if (i + 1) % batch_size == 0:
                    self.build_index()
                    print(f"Saved progress: {successfully_added} images added so far")
                    
            except Exception as e:
                print(f"Error processing image {url}: {e}")
                continue
        
        # Final save
        self.build_index()
        print(f"Successfully added {successfully_added}/{len(image_urls)} images to the index")
        return successfully_added
    
    def add_images_from_folder(self, folder_path, batch_size=10, allowed_exts=(".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        """Add images from a local folder to the index, avoiding duplicates."""

        successfully_added = 0
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(allowed_exts)]
       
        # Avoid duplicates: check if file path is already in metadata
        existing_paths = set()
        for meta in self.image_metadata:
            if 'file_path' in meta:
                existing_paths.add(meta['file_path'])
        new_files = [f for f in all_files if f not in existing_paths]
        for i, file_path in enumerate(new_files):
            try:
                print(f"Processing image {i+1}/{len(new_files)}: {file_path}")
                image = Image.open(file_path).convert("RGB")
                embedding = self.embed_img(image)
                metadata = {
                    'file_path': file_path,
                    'added_at': str(i),
                    'original_index': i
                }
                self.add_vector(embedding, metadata)
                successfully_added += 1
                if (i + 1) % batch_size == 0:
                    self.build_index()
                    print(f"Saved progress: {successfully_added} images added so far")
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                continue
        self.build_index()
        print(f"Successfully added {successfully_added}/{len(new_files)} new images to the index from folder {folder_path}")
        return successfully_added

def main(clear_index:bool = False, add_new_images:bool = False, folder_path = None, img_query_path : str = '/Users/varamana/Desktop/Wiki/IMG-20231026-WA0025.jpg'):
    # Initialize pipeline (will automatically load existing index if available)
    pipeline = ImageToImageSearch(image_dir="/Users/varamana/Desktop/Wiki")
    print("Pipeline initialized successfully.")
    
    # Print current index info
    info = pipeline.get_index_info()
    print(f"Current index info: {info}")

    if clear_index:
        print("Clearing existing index...")
        pipeline.clear_index()

    if pipeline.image_dir is None:
        pipeline.image_dir = "/Users/varamana/Desktop/Wiki"
    print(f"Adding images from local folder: {pipeline.image_dir}")

    
    if add_new_images:
        print("Index has {pipeline.index.ntotal} images after adding new images.")


    # Option 3: Search for similar images
    if pipeline.index.ntotal > 0:
        print("\nPerforming similarity search...")
        # Use a query image from the folder (first file in metadata with 'file_path', else first file in folder)
        if img_query_path:
            try:
                query_image = Image.open(img_query_path).convert("RGB")
                # Search for similar images
                results = pipeline.img_search(query_image, top_k=3)
                print(f"Top 3 similar images to the query image ({img_query_path}):")
                for i, result in enumerate(results):
                    print(f"{i+1}. Distance: {result['distance']:.4f}")
                    print(f"   File: {result['metadata'].get('file_path', 'N/A')}")
                    print(f"   Index: {result['index']}")
                    Image.open(result['metadata'].get('file_path', 'N/A')).show(title=f"Result {i+1}")

                query_image.show(title="Query Image")
            except Exception as e:
                print(f"Error with query image: {e}")
        else:
            print("No local image found for similarity search.")
    else:
        print("Index is empty. Add images before performing a search.")
    # Final save to ensure persistence
    pipeline.build_index()
    
    # Print final stats
    final_info = pipeline.get_index_info()
    print(f"Final index info: {final_info}")

if __name__ == "__main__":
    main()    

    