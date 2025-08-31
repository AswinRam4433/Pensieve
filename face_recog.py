import os
import pickle
import numpy as np
import faiss
import face_recognition
from PIL import Image
from typing import List, Dict, Tuple, Optional


class FaceIndexer:
    """A clean, efficient face recognition and indexing system."""
    
    def __init__(self, image_dir: str = "images", index_path: str = "faces_new.index", 
                 meta_path: str = "metadata.pkl"):
        self.image_dir = image_dir
        self.index_path = index_path
        self.meta_path = meta_path
        self.embeddings_path = meta_path.replace('.pkl', '_embeddings.npy')
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Quality thresholds
        self.min_face_size = 50  # Minimum face size in pixels
        self.distance_threshold = 0.6  # Face similarity threshold (lower = more similar)
    
    def _is_valid_image(self, filename: str) -> bool:
        """Check if file is a supported image format."""
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)
    
    def _validate_face_quality(self, face_location: Tuple, image_shape: Tuple) -> bool:
        """Validate if detected face meets quality requirements."""
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        # Check minimum face size
        if face_width < self.min_face_size or face_height < self.min_face_size:
            return False
        
        # Check if face is not too close to image borders (partial faces)
        margin = 10
        height, width = image_shape[:2]
        if (left < margin or top < margin or 
            right > width - margin or bottom > height - margin):
            return False
            
        return True
    
    def build_index(self) -> Tuple[faiss.Index, Dict]:
        """Build FAISS index from images in directory."""
        print(f"Building face index from directory: {self.image_dir}")
        
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory '{self.image_dir}' does not exist!")
        
        embeddings = []
        face_ids = []
        face_to_photos = {}  # {face_id: (filename, face_index, face_location)}
        files_to_faces = {}  # {filename: [face_ids]}
        
        image_files = [f for f in os.listdir(self.image_dir) if self._is_valid_image(f)]
        total_files = len(image_files)
        
        if total_files == 0:
            raise ValueError(f"No valid image files found in {self.image_dir}")
        
        print(f"Processing {total_files} image files...")
        
        for i, filename in enumerate(image_files):
            if i % 50 == 0:  # Progress every 50 files
                print(f"Progress: {i}/{total_files} files processed")
            
            file_path = os.path.join(self.image_dir, filename)
            
            try:
                # Load image and detect faces
                image = face_recognition.load_image_file(file_path)
                face_locations = face_recognition.face_locations(image, model="hog")  # Use HOG for speed
                
                if not face_locations:
                    continue
                
                # Filter out low-quality faces
                valid_locations = [loc for loc in face_locations 
                                 if self._validate_face_quality(loc, image.shape)]
                
                if not valid_locations:
                    continue
                
                # Get face encodings for valid faces only
                face_encodings = face_recognition.face_encodings(image, valid_locations)
                
                if not face_encodings:
                    continue
                
                files_to_faces[filename] = []
                
                for face_idx, (encoding, location) in enumerate(zip(face_encodings, valid_locations)):
                    # Create unique face ID
                    face_id = f"{filename}_face_{face_idx}"
                    
                    # Store data
                    embeddings.append(encoding)
                    face_ids.append(face_id)
                    files_to_faces[filename].append(face_id)
                    face_to_photos[face_id] = (filename, face_idx, location)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if len(embeddings) == 0:
            raise ValueError("No valid face embeddings found in the directory!")
        
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"Successfully extracted {len(embeddings)} face embeddings from {len(files_to_faces)} images.")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index with cosine similarity (Inner Product after normalization)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embeddings)
        
        # Save everything
        self._save_index_and_metadata(index, embeddings, face_ids, files_to_faces, face_to_photos)
        
        metadata = {
            "face_ids": face_ids,
            "files_to_faces": files_to_faces,
            "face_to_photos": face_to_photos,
            "total_faces": len(face_ids),
            "total_images": len(files_to_faces)
        }
        
        return index, metadata
    
    def _save_index_and_metadata(self, index: faiss.Index, embeddings: np.ndarray, 
                                face_ids: List[str], files_to_faces: Dict, 
                                face_to_photos: Dict) -> None:
        """Save index, embeddings, and metadata to disk."""
        # Save FAISS index
        faiss.write_index(index, self.index_path)
        
        # Save embeddings
        np.save(self.embeddings_path, embeddings)
        
        # Save metadata
        metadata = {
            "face_ids": face_ids,
            "files_to_faces": files_to_faces,
            "face_to_photos": face_to_photos,
            "total_faces": len(face_ids),
            "total_images": len(files_to_faces),
            "embeddings_path": self.embeddings_path
        }
        
        with open(self.meta_path, "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Saved index to {self.index_path}")
        print(f"✅ Saved metadata to {self.meta_path}")
        print(f"✅ Saved embeddings to {self.embeddings_path}")
    
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
        
        print(f"✅ Loaded index with {metadata['total_faces']} faces from {metadata['total_images']} images")
        return index, metadata, embeddings
    
    def search_similar_faces(self, query_image_path: str, index: faiss.Index, 
                           metadata: Dict, k: int = 5) -> List[Dict]:
        """Search for faces similar to those in the query image."""
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image {query_image_path} not found.")
        
        try:
            # Load and process query image
            image = face_recognition.load_image_file(query_image_path)
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                print("❌ No faces detected in query image")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                print("❌ Could not encode faces in query image")
                return []
            
            print(f"🔍 Found {len(face_encodings)} face(s) in query image")
            
            all_results = []
            face_ids = metadata["face_ids"]
            face_to_photos = metadata["face_to_photos"]
            
            for face_idx, encoding in enumerate(face_encodings):
                # Normalize query encoding for cosine similarity
                query_embedding = np.array([encoding], dtype=np.float32)
                faiss.normalize_L2(query_embedding)
                
                # Search in index
                similarities, indices = index.search(query_embedding, k * 2)  # Get more results to filter
                
                # Filter results by similarity threshold and remove duplicates
                valid_results = []
                seen_photos = set()
                
                for j, idx in enumerate(indices[0]):
                    if idx == -1:  # FAISS returns -1 for invalid indices
                        continue
                    
                    similarity = float(similarities[0][j])
                    # Convert similarity to distance (higher similarity = lower distance)
                    distance = 1.0 - similarity
                    
                    # Apply similarity threshold
                    if distance > self.distance_threshold:
                        continue
                    
                    face_id = face_ids[idx]
                    photo_name, face_index, face_location = face_to_photos[face_id]
                    
                    # Avoid duplicate photos (optional - remove if you want all faces)
                    if photo_name in seen_photos:
                        continue
                    seen_photos.add(photo_name)
                    
                    valid_results.append({
                        "face_id": face_id,
                        "photo_path": os.path.join(self.image_dir, photo_name),
                        "photo_name": photo_name,
                        "face_index": face_index,
                        "distance": distance,
                        "similarity": similarity,
                        "face_location": face_location
                    })
                    
                    if len(valid_results) >= k:  # Stop when we have enough results
                        break
                
                all_results.append({
                    "query_face_index": face_idx,
                    "query_face_location": face_locations[face_idx],
                    "similar_faces": valid_results
                })
            
            return all_results
            
        except Exception as e:
            print(f"Error processing query image: {e}")
            return []
    
    def get_face_clusters(self, embeddings: np.ndarray, num_clusters: int = 10) -> np.ndarray:
        """Cluster face embeddings using K-means."""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings provided for clustering")
        
        print(f"🤖 Clustering {len(embeddings)} faces into {num_clusters} clusters...")
        
        # Ensure embeddings are normalized
        normalized_embeddings = embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # Perform K-means clustering
        dimension = embeddings.shape[1]
        kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=False)
        kmeans.train(np.ascontiguousarray(normalized_embeddings))
        
        # Assign each face to a cluster
        _, cluster_assignments = kmeans.assign(np.ascontiguousarray(normalized_embeddings))
        
        return cluster_assignments.flatten()
    
    def display_results(self, results: List[Dict], show_images: bool = False) -> None:
        """Display search results in a formatted way."""
        if not results:
            print("No similar faces found.")
            return
        
        for query_result in results:
            query_face_idx = query_result["query_face_index"]
            similar_faces = query_result["similar_faces"]
            
            print(f"\n🔍 Results for Query Face #{query_face_idx}:")
            print(f"Found {len(similar_faces)} similar face(s)")
            
            for i, result in enumerate(similar_faces, 1):
                similarity_percent = result["similarity"] * 100
                print(f"  {i}. {result['photo_name']} (Face #{result['face_index']})")
                print(f"     Similarity: {similarity_percent:.1f}% | Distance: {result['distance']:.3f}")
                print(f"     Path: {result['photo_path']}")
                
                if show_images:
                    try:
                        Image.open(result['photo_path']).show()
                    except Exception as e:
                        print(f"     Could not display image: {e}")
                print()


def main():
    """Main execution function."""
    # Configuration
    IMAGE_DIR = "/Users/varamana/Desktop/Wiki"  # Update this path
    QUERY_IMAGE = "/Users/varamana/Desktop/Wiki/IMG-20220524-WA0026.jpg"  # Update this path
    
    # Initialize the face indexer
    indexer = FaceIndexer(image_dir=IMAGE_DIR)
    
    # Check if index exists, if not build it
    if not os.path.exists(indexer.index_path):
        print("🔧 Index not found. Building new index...")
        index, metadata = indexer.build_index()
    else:
        print("📁 Loading existing index...")
        index, metadata, embeddings = indexer.load_index()
    
    # Search for similar faces
    print(f"\n🔍 Searching for faces similar to: {QUERY_IMAGE}")
    results = indexer.search_similar_faces(QUERY_IMAGE, index, metadata, k=5)
    
    # Display results
    indexer.display_results(results)
    
    # Optional: Clustering example
    # if embeddings is not None:
    #     print("\n🤖 Performing face clustering...")
    #     cluster_assignments = indexer.get_face_clusters(embeddings, num_clusters=5)
    #     
    #     # Show cluster statistics
    #     unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
    #     print("Cluster distribution:")
    #     for cluster_id, count in zip(unique_clusters, counts):
    #         print(f"  Cluster {cluster_id}: {count} faces")


if __name__ == "__main__":
    main()