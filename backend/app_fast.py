# from fastapi.responses import FileResponse
# from fastapi import Query
# import os
# from fastapi import FastAPI, UploadFile, File, Form, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Optional
# import shutil
# import tempfile
# import threading

# from config import SystemConfig

# app = FastAPI(
#     title="Integrated Image Search API",
#     description="Unified API for face, text, and image-to-image search with lazy loading.",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variable to hold the pipeline (loaded lazily)
# pipeline = None
# pipeline_lock = threading.Lock()

# def get_pipeline():
#     """Lazy load the pipeline only when first needed"""
#     global pipeline
#     if pipeline is None:
#         with pipeline_lock:
#             if pipeline is None:  # Double-checked locking
#                 print("Initializing pipeline for the first time...")
#                 from integrated import IntegratedPipeline
#                 IMAGE_DIR = SystemConfig.my_img_directory
#                 pipeline = IntegratedPipeline(image_dir=IMAGE_DIR)
#                 print("Pipeline initialization completed!")
#     return pipeline

# @app.get("/", tags=["Health"])
# def health_check():
#     """Quick health check without loading models"""
#     return {"status": "healthy", "pipeline_loaded": pipeline is not None}

# @app.get("/info", tags=["Info"])
# def get_index_info():
#     """
#     Get information about all indexes (face, text, image-to-image).
#     This will trigger pipeline loading if not already loaded.
#     """
#     try:
#         p = get_pipeline()
#         info = p.get_index_info()
#         return info
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# # Serve images by path (for frontend display)
# @app.get('/image', tags=["Images"])
# def get_image(path: str = Query(..., description='Absolute or relative image path')):
#     """
#     Serve an image file by absolute or relative path (restricted to IMAGE_DIR).
#     """
#     import os
#     base_dir = os.path.abspath(SystemConfig.my_img_directory)
#     abs_path = os.path.abspath(path)
#     if not abs_path.startswith(base_dir):
#         return JSONResponse(status_code=403, content={'error': 'Access denied'})
#     if not os.path.exists(abs_path):
#         return JSONResponse(status_code=404, content={'error': 'Image not found'})
#     return FileResponse(abs_path)

# @app.get("/faces", response_model=None, tags=["Faces"])
# def list_faces(with_thumbnails: bool = False, thumbnail_size: int = 64):
#     """
#     List all faces in the index for dropdowns, optionally with base64 thumbnails.
#     """
#     try:
#         p = get_pipeline()
#         faces = p.list_faces(with_thumbnails=with_thumbnails, thumbnail_size=thumbnail_size)
#         return {"faces": faces}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/scan", tags=["Admin"])
# def scan_repo():
#     """
#     Build all indexes from scratch. This may take a while for large datasets.
#     """
#     try:
#         p = get_pipeline()
#         p.scan_repo()
#         return {"status": "success", "message": "All indexes rebuilt."}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/search", tags=["Search"])
# async def unified_search(
#     request: Request,
#     face_image: Optional[UploadFile] = File(None, description="Image file for face search (optional)"),
#     query_image: Optional[UploadFile] = File(None, description="Image file for image-to-image search (optional)"),
#     text: Optional[str] = Form(None, description="Text query for text-based search (optional)"),
#     natural_query: Optional[str] = Form(None, description="Natural language query (optional)"),
#     max_results: Optional[int] = Form(10, description="Maximum number of results to return")
# ):
#     """
#     Unified search endpoint.
#     """
#     query = {}
#     temp_files = []
    
#     try:
#         p = get_pipeline()  # This will load the pipeline if needed

#         # Handle natural language query parsing
        

#         # Handle face image upload
#         if face_image is not None:
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#                     shutil.copyfileobj(face_image.file, tmp)
#                     tmp_path = tmp.name
#                     query["face_image_path"] = tmp_path
#                     temp_files.append(tmp_path)
#             except Exception as e:
#                 return JSONResponse(status_code=400, content={"error": f"Failed to process face image: {e}"})

#         # Handle query image upload
#         if query_image is not None:
#             try:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#                     shutil.copyfileobj(query_image.file, tmp)
#                     tmp_path = tmp.name
#                     query["query_image_path"] = tmp_path
#                     temp_files.append(tmp_path)
#             except Exception as e:
#                 return JSONResponse(status_code=400, content={"error": f"Failed to process query image: {e}"})

#         # Handle text query
#         if text is not None:
#             query["text"] = text
        
#         if max_results is None:
#             max_results = 5

#         # Run unified search
#         results = p.unified_search(query, max_results)
#         return JSONResponse(content=results)
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         # Clean up temp files
#         for f in temp_files:
#             try:
#                 os.remove(f)
#             except Exception:
#                 pass

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
