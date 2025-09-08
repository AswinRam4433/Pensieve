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
    """
    Integrates FaceIndexer, CLIPImageSearcher, and ImageToImageSearch for unified search.
    Supports: face search, text search, image-to-image search, and combined queries.
    """
    def __init__(self, image_dir: str) -> None:
        self.image_dir = image_dir
        self.face_indexer = FaceIndexer(image_dir=image_dir)
        self.clip_searcher = CLIPImageSearcher(image_dir=image_dir)
        self.img2img_searcher = ImageToImageSearch(image_dir=image_dir)

        # Index/cache
        self.face_index = None
        self.face_metadata = None
        self.clip_index = None
        self.clip_metadata = None
        self.img2img_index = None
        self.img2img_metadata = None

        self._load_all_indexes()

    def _load_all_indexes(self):
        """Load all indexes and metadata into memory."""
        try:
            self.face_index, self.face_metadata, self.face_embeddings = self.face_indexer.load_index()
        except Exception as e:
            print(f"Face index not loaded: {e}")
            self.face_index, self.face_metadata, self.face_embeddings = None, None, None
        try:
            self.clip_index, self.clip_metadata, self.clip_embeddings = self.clip_searcher.load_index()
        except Exception as e:
            print(f"CLIP index not loaded: {e}")
            self.clip_index, self.clip_metadata, self.clip_embeddings = None, None, None
        try:
            self.img2img_index = self.img2img_searcher.index
            self.img2img_metadata = self.img2img_searcher.image_metadata
        except Exception as e:
            print(f"Image2Image index not loaded: {e}")
            self.img2img_index, self.img2img_metadata = None, None

    def scan_repo(self):
        """Build all indexes from scratch."""
        self.face_indexer.build_index()
        self.clip_searcher.build_index()
        self.img2img_searcher.build_index()
        self._load_all_indexes()

    def get_index_info(self):
        """Get info about all indexes."""
        info = {
            "face_index": self.face_indexer.get_face_clusters(self.face_embeddings) if self.face_embeddings is not None else "N/A",
            "clip_index": self.clip_searcher.get_index_stats(self.clip_metadata) if self.clip_metadata else "N/A",
            "img2img_index": self.img2img_searcher.get_index_info() if self.img2img_index else "N/A"
        }
        return info

    def list_faces(self, with_thumbnails: bool = False, thumbnail_size: int = 64) -> list:
        """
        List all faces in the index for dropdowns, optionally with thumbnails (base64 encoded).
        """
        if not self.face_metadata:
            return []
        faces = []
        for face_id in self.face_metadata["face_ids"]:
            photo_name, face_idx, face_location = self.face_metadata["face_to_photos"][face_id]
            face_info = {
                "face_id": face_id,
                "photo_name": photo_name,
                "face_index": face_idx,
                "face_location": face_location
            }
            if with_thumbnails:
                try:
                    thumb = self.get_face_thumbnail(photo_name, face_location, size=thumbnail_size)
                    face_info["thumbnail"] = thumb
                except Exception:
                    face_info["thumbnail"] = None
            faces.append(face_info)
        return faces

    def get_face_thumbnail(self, photo_name: str, face_location: tuple, size: int = 64) -> str:
        """
        Get a base64-encoded thumbnail of a face given photo name and face location.
        """
        import base64
        from io import BytesIO
        img_path = os.path.join(self.image_dir, photo_name)
        image = Image.open(img_path).convert("RGB")
        top, right, bottom, left = face_location
        face_img = image.crop((left, top, right, bottom)).resize((size, size))
        buffer = BytesIO()
        face_img.save(buffer, format="JPEG")
        thumb_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return thumb_b64

    def unified_search(self, query: dict, top_k: int = 5):
        """
        Accepts a dict query, e.g.:
        {
            "face_image_path": "...",   # optional, for face search
            "text": "...",              # optional, for text search
            "query_image_path": "...",  # optional, for image-to-image search
            "require_all": True/False   # if True, return intersection; else, union
        }
        Returns: dict with results from each search type and optionally combined.
        """
        results = {}
        require_all = query.get("require_all", False)

        # Face search
        if "face_image_path" in query and query["face_image_path"] and self.face_index and self.face_metadata:
            face_results = self.face_indexer.search_similar_faces(
                query["face_image_path"], self.face_index, self.face_metadata, k=top_k
            )
            results["face"] = face_results

        # Text search
        if "text" in query and query["text"] and self.clip_index and self.clip_metadata:
            text_results = self.clip_searcher.search_by_text(
                query["text"], self.clip_index, self.clip_metadata, k=top_k
            )
            results["text"] = text_results

        # Image-to-image search
        if "query_image_path" in query and query["query_image_path"] and self.img2img_index and self.img2img_metadata:
            try:
                img = Image.open(query["query_image_path"]).convert("RGB")
                img_results = self.img2img_searcher.img_search(img, top_k=top_k)
                results["image"] = img_results
            except Exception as e:
                results["image"] = f"Error: {e}"

        # Combine/intersect results if required
        if require_all and results:
            # Find intersection of image paths across all result types
            sets = []
            if "face" in results:
                face_paths = set(
                    r["photo_path"] for q in results["face"] for r in q.get("similar_faces", [])
                )
                sets.append(face_paths)
            if "text" in results:
                text_paths = set(r["image_path"] for r in results["text"])
                sets.append(text_paths)
            if "image" in results and isinstance(results["image"], list):
                img_paths = set(r["metadata"].get("file_path") for r in results["image"] if "metadata" in r)
                sets.append(img_paths)
            if sets:
                intersection = set.intersection(*sets) if sets else set()
                results["intersection"] = list(intersection)
        return results

    # def parse_natural_query(self, query_str: str):
    #     """
    #     Parse a natural language query like:
    #     "has to have face A, matches this description"
    #     Returns a dict suitable for unified_search.
    #     """
    #     # Simple heuristic parser (can be improved with NLP)
    #     query = {}
    #     if "face" in query_str.lower():
    #         # Expecting a face image path to be provided separately in API
    #         query["face_image_path"] = None
    #     if "description" in query_str.lower() or "text" in query_str.lower():
    #         # Expecting a text description to be provided separately in API
    #         query["text"] = None
    #     if "image" in query_str.lower() or "similar to" in query_str.lower():
    #         query["query_image_path"] = None
    #     # Add more parsing logic as needed
    #     return query
