from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Test API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is working!"}

@app.get("/test-volume")
def test_volume():
    """Test if the volume mount is working"""
    image_dir = "<<YourSampleImageDir>>"
    try:
        if os.path.exists(image_dir):
            files = os.listdir(image_dir)[:10]  # List first 10 files
            return {
                "status": "success",
                "image_dir": image_dir,
                "exists": True,
                "sample_files": files
            }
        else:
            return {
                "status": "error",
                "image_dir": image_dir,
                "exists": False,
                "message": "Image directory not found"
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
