#!/usr/bin/env python3
"""
Pre-download models to cache them in Docker image
"""

try:
    print("Starting model download...")
    
    from transformers import AutoProcessor, AutoModel
    import torch
    
    print("Downloading SigLIP models...")
    processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-384')
    model = AutoModel.from_pretrained('google/siglip-base-patch16-384')
    print("SigLIP models downloaded successfully")
    
    # Test face recognition import
    print("Testing face recognition...")
    import face_recognition
    print("Face recognition is ready")
    
    print("All models cached successfully!")
    
except Exception as e:
    print(f"Model download failed: {e}")
    # Don't fail the build, just log the error
    import sys
    sys.exit(0)
