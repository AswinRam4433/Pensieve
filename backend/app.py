from fastapi.responses import FileResponse
from fastapi import Query
import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import shutil
import tempfile

from integrated import IntegratedPipeline
from config import SystemConfig

app = FastAPI(
    title="Integrated Image Search API",
    description="Unified API for text and image-to-image search.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMAGE_DIR = SystemConfig.my_img_directory
pipeline = IntegratedPipeline(image_dir=IMAGE_DIR)

# Serve images by path (for frontend display)
@app.get('/image', tags=["Images"])
def get_image(path: str = Query(..., description='Absolute or relative image path')):
    """
    Serve an image file by absolute or relative path (restricted to IMAGE_DIR).
    """
    import os
    base_dir = os.path.abspath(IMAGE_DIR)
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(base_dir):
        return JSONResponse(status_code=403, content={'error': 'Access denied'})
    if not os.path.exists(abs_path):
        return JSONResponse(status_code=404, content={'error': 'Image not found'})
    return FileResponse(abs_path)

@app.get("/faces", response_model=None, tags=["Faces"])
def list_faces(with_thumbnails: bool = False, thumbnail_size: int = 64):
    """
    List all faces in the index for dropdowns, optionally with base64 thumbnails.
    """
    try:
        faces = pipeline.list_faces(with_thumbnails=with_thumbnails, thumbnail_size=thumbnail_size)
        return {"faces": faces}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/scan", tags=["Admin"])
def scan_repo():
    """
    Build all indexes from scratch. This may take a while for large datasets.
    """
    try:
        pipeline.scan_repo()
        return {"status": "success", "message": "All indexes rebuilt."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/info", tags=["Info"])
def get_index_info():
    """
    Get information about all indexes (face, text, image-to-image).
    """
    try:
        info = pipeline.get_index_info()
        return info
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/search", tags=["Search"])
async def unified_search(
    request: Request,
    face_image: Optional[UploadFile] = File(None, description="Image file for face search (optional)"),
    query_image: Optional[UploadFile] = File(None, description="Image file for image-to-image search (optional)"),
    text: Optional[str] = Form(None, description="Text query for text-based search (optional)"),
    max_results: Optional[int] = Form(10, description="Maximum number of results to return")
):
    """
    Unified search endpoint.
    Accepts:
    - face_image: image file for face search (optional)
    - query_image: image file for image-to-image search (optional)
    - text: text query for text search (optional)
    - require_all: if True, returns intersection of results (optional)
    - natural_query: natural language query (optional, e.g. "has to have face A, matches this description")
    Returns: JSON with results from each search type and optionally combined.
    """
    query = {}
    temp_files = []

    # Handle face image upload
    if face_image is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(face_image.file, tmp)
                tmp_path = tmp.name
                query["face_image_path"] = tmp_path
                temp_files.append(tmp_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Failed to process face image: {e}"})

    # Handle query image upload
    if query_image is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                shutil.copyfileobj(query_image.file, tmp)
                tmp_path = tmp.name
                query["query_image_path"] = tmp_path
                temp_files.append(tmp_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Failed to process query image: {e}"})

    # Handle text query
    if text is not None:
        query["text"] = text
    
    if max_results is None:
        max_results = 5
    
    # Run search
    try:
       
        results = pipeline.unified_search(query, max_results)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
