import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
from typing import List, Tuple, Dict, Optional
import faiss
from pathlib import Path
import json
from datetime import datetime
from config import SystemConfig
from utils.custom_logging import custom_logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class CLIPImageSearcher:
    """A CLIP-based image search system with persistent storage."""
    
    def __init__(self, image_dir: str = "images", model_name: str = "ViT-B/32",
                 index_path: str = "clip_index.faiss", meta_path: str = "clip_metadata.pkl"):
        self.image_dir = Path(image_dir)
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path
        self.embeddings_path = meta_path.replace('.pkl', '_embeddings.npy')
        
        self.supported_formats = SystemConfig.supported_formats
        
        self.device = SystemConfig.device
        custom_logger.info(f"Using device: {self.device}")
        
        ## GAI START
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode --> Missed this out
        ## GAI END
        
        self.similarity_threshold = 0.1  
        self.batch_size = 32  
    
    def _is_valid_image(self, filepath: Path) -> bool:
        """Check if file is a supported image format."""
        return filepath.suffix.lower() in self.supported_formats
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Safely load and preprocess an image."""
        try:
            image = Image.open(image_path).convert('RGB')  # Ensure RGB format
            return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _process_images_in_batches(self, image_paths: List[Path]) -> Tuple[np.ndarray, List[str]]:
        """Process images in batches for memory efficiency."""
        embeddings = []
        valid_paths = []
        
        total_images = len(image_paths)
        print(f"📸 Processing {total_images} images in batches of {self.batch_size}...")
        
        for i in range(0, total_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # Load batch
            for path in batch_paths:
                processed_image = self._load_and_preprocess_image(path)
                if processed_image is not None:
                    batch_images.append(processed_image)
                    batch_valid_paths.append(str(path.relative_to(self.image_dir)))
            
            if not batch_images:
                continue
            
            # Process batch
            batch_tensor = torch.cat(batch_images, dim=0)
            
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features = batch_features.cpu().numpy()
            
            embeddings.append(batch_features)
            valid_paths.extend(batch_valid_paths)
            
            processed = min(i + self.batch_size, total_images)
            print(f"Progress: {processed}/{total_images} images processed")
        
        if not embeddings:
            raise ValueError("No valid images found to process!")
        

        ## GAI START
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Normalize for cosine similarity
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        ## GAI END

        return all_embeddings, valid_paths
    
    def build_index(self, force_rebuild: bool = False) -> Tuple[faiss.Index, Dict]:
        """Build CLIP image index with persistent storage."""
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print("📁 Index already exists. Use force_rebuild=True to rebuild.")
            return self.load_index()
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory '{self.image_dir}' does not exist!")
        
        # Get all image files
        image_files = [f for f in self.image_dir.rglob("*") 
                      if f.is_file() and self._is_valid_image(f)]
        
        if not image_files:
            raise ValueError(f"No valid images found in {self.image_dir}")
        
        custom_logger.info(f"Found {len(image_files)} images for indexing")
        
        # Process images and extract embeddings
        embeddings, valid_paths = self._process_images_in_batches(image_files)
        
        
        custom_logger.info(f"Successfully processed {len(valid_paths)} images")
        custom_logger.info(f"Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index with cosine similarity
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # Save everything
        self._save_index_and_metadata(index, embeddings, valid_paths)
        
        metadata = {
            "image_paths": valid_paths,
            "total_images": len(valid_paths),
            "model_name": self.model_name,
            "embedding_dim": dimension,
            "created_at": datetime.now().isoformat(),
            "image_dir": str(self.image_dir)
        }
        
        return index, metadata
    
    def _save_index_and_metadata(self, index: faiss.Index, embeddings: np.ndarray, 
                                image_paths: List[str]) -> None:
        """Save index, embeddings, and metadata to disk."""
        # Save FAISS index
        faiss.write_index(index, self.index_path)
        
        # Save embeddings
        np.save(self.embeddings_path, embeddings)
        
        # Save metadata
        metadata = {
            "image_paths": image_paths,
            "total_images": len(image_paths),
            "model_name": self.model_name,
            "embedding_dim": embeddings.shape[1],
            "created_at": datetime.now().isoformat(),
            "image_dir": str(self.image_dir),
            "embeddings_path": self.embeddings_path
        }
        
        with open(self.meta_path, "wb") as f:
            pickle.dump(metadata, f)
        

        custom_logger.info(f"Saved index to {self.index_path}")
        custom_logger.info(f"Saved metadata to {self.meta_path}")
        custom_logger.info(f"Saved embeddings to {self.embeddings_path}")
    
    def load_index(self) -> Tuple[faiss.Index, Dict, Optional[np.ndarray]]:
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file {self.index_path} not found. Run build_index() first.")
        
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file {self.meta_path} not found.")
        
        # Load index
        index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        # Load embeddings if available
        embeddings = None
        if os.path.exists(self.embeddings_path):
            embeddings = np.load(self.embeddings_path)
        
        
        custom_logger.info(f"Loaded index with {metadata['total_images']} images")
        custom_logger.info(f"Model: {metadata['model_name']}, Created: {metadata['created_at']}")
        
        return index, metadata, embeddings
    
    def search_by_text(self, query: str, index: faiss.Index, metadata: Dict, 
                      k: int = 5) -> List[Dict]:
        """Search for images using text description."""
        custom_logger.info(f"Text search query: {query}")
        
        try:
            # Encode text query
            with torch.no_grad():
                text_tokens = clip.tokenize([query]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features.cpu().numpy()
                text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
            # Search in index
            similarities, indices = index.search(text_features.astype('float32'), k * 2)
            
            # Filter and format results
            results = []
            image_paths = metadata["image_paths"]
            
            for j, idx in enumerate(indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                similarity = float(similarities[0][j])
                
                # Apply similarity threshold
                if similarity < self.similarity_threshold:
                    continue
                
                image_path = image_paths[idx]
                full_path = self.image_dir / image_path
                
                results.append({
                    "image_path": str(full_path),
                    "relative_path": image_path,
                    "similarity": similarity,
                    "score_percentage": similarity * 100,
                    "query": query,
                    "search_type": "text"
                })
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error in text search: {e}")
            return []
    
    
    def display_results(self, results: List[Dict], show_images: bool = False) -> None:
        """Display search results in a formatted way."""
        if not results:
            custom_logger.warn("No results found.")
            return
        
        search_type = results[0]["search_type"]
        query = results[0]["query"]

        custom_logger(f"\nTop {len(results)} results for {search_type} search qeury:{query}")
        
        for i, result in enumerate(results, 1):
            score_percent = result["score_percentage"]
            print(f"{i:2d}. {result['relative_path']}")
            print(f"    Similarity: {score_percent:.1f}% | Score: {result['similarity']:.4f}")
            print(f"    Full path: {result['image_path']}")
            
            if show_images:
                try:
                    Image.open(result['image_path']).show()
                except Exception as e:
                    print(f"Could not display image: {e}")
            print()
    
    def get_index_stats(self, metadata: Dict) -> Dict:
        """Get statistics about the current index."""
        return {
            "total_images": metadata["total_images"],
            "model_name": metadata["model_name"],
            "embedding_dimension": metadata["embedding_dim"],
            "created_at": metadata["created_at"],
            "image_directory": metadata["image_dir"],
            "index_size_mb": os.path.getsize(self.index_path) / (1024 * 1024) if os.path.exists(self.index_path) else 0
        }
    
    def export_results_to_json(self, results: List[Dict], output_path: str) -> None:
        """Export search results to JSON file."""
        export_data = {
            "search_timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {output_path}")
        


def main():
    """Main execution function with examples."""
    # Configuration
    IMAGE_DIR = SystemConfig.my_img_directory  # Update this path
    QUERY_IMAGE = SystemConfig.sample_query_img_path  # Update this path
    
    # Initialize the searcher
    searcher = CLIPImageSearcher(image_dir=IMAGE_DIR)
    
    try:
        # Check if index exists, build if not
        if not os.path.exists(searcher.index_path):
            print("Building new CLIP index...")
            index, metadata = searcher.build_index()
        else:
            print("Loading existing index...")
            index, metadata, embeddings = searcher.load_index()
        
        # Display index statistics
        stats = searcher.get_index_stats(metadata)
        print(f"\nIndex Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        
        # Example 1: Text-based search
        text_queries = [
            "a boy wearing a mask",
            "people smiling",
            "outdoor scenery",
            "group photo"
        ]
        
        for query in text_queries[:2]:  # Test first 2 queries
            print(f"\nTEXT SEARCH")
            results = searcher.search_by_text(query, index, metadata, k=3)
            searcher.display_results(results)
        
        
        else:
            print(f"Query image {QUERY_IMAGE} not found. Skipping image search.")
        
        # Example 3: Find potential duplicates
        # duplicate_groups = searcher.find_duplicate_images(index, metadata)
        # if duplicate_groups:
        #     print(f"\n🔄 Potential Duplicate Groups:")
        #     for i, group in enumerate(duplicate_groups[:3], 1):  # Show first 3 groups
        #         print(f"Group {i}:")
        #         for img in group:
        #             print(f"  - {img}")
        
    except Exception as e:
        print(f"Error: {e}")


# Utility functions for advanced use cases
def batch_text_search(searcher: CLIPImageSearcher, queries: List[str], 
                     index: faiss.Index, metadata: Dict, k: int = 5) -> Dict[str, List[Dict]]:
    """Perform batch text searches for multiple queries."""
    results = {}
    for query in queries:
        results[query] = searcher.search_by_text(query, index, metadata, k)
    return results


def find_images_with_multiple_concepts(searcher: CLIPImageSearcher, concepts: List[str],
                                      index: faiss.Index, metadata: Dict, 
                                      k: int = 10) -> List[Dict]:
    """Find images that match multiple concepts (intersection of results)."""
    all_results = {}
    
    # Search for each concept
    for concept in concepts:
        results = searcher.search_by_text(concept, index, metadata, k=k*2)
        all_results[concept] = {r["relative_path"]: r for r in results}
    
    # Find intersection
    if not all_results:
        return []
    
    # Start with results from first concept
    common_images = set(all_results[concepts[0]].keys())
    
    # Intersect with other concepts
    for concept in concepts[1:]:
        common_images &= set(all_results[concept].keys())
    
    # Create combined results with average similarity
    combined_results = []
    for img_path in common_images:
        similarities = [all_results[concept][img_path]["similarity"] for concept in concepts]
        avg_similarity = np.mean(similarities)
        
        result = all_results[concepts[0]][img_path].copy()
        result["similarity"] = avg_similarity
        result["score_percentage"] = avg_similarity * 100
        result["matched_concepts"] = concepts
        result["individual_scores"] = {concept: all_results[concept][img_path]["similarity"] 
                                     for concept in concepts}
        
        combined_results.append(result)
    
    # Sort by average similarity
    combined_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return combined_results[:k]


if __name__ == "__main__":
    main()