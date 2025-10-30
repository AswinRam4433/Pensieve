# Pensieve

A privacy-focused, all-local photo organization and search application. Pensieve allows you to search your entire photo collection entirely on-device, without cloud uploads. This app allows image-to-image similarity search and text-to-image search.

## How does it work

This app makes use of React and Typescript for frontend and Python with Fastapi for the backend. For the image-to-image similarity search, we use Google's SigLIP model. The embeddings are saved in FAISS(siglip_index.faiss) and they have metadata stored in JSON format to list the file paths for each image.
The text-to-image model runs using CLIP. The embeddings are stored in FAISS(clip_index.faiss).
Both of these are open-source modles and run locally on the user's machine and can easily run on CPU.

## Pre-requisites & Setup

### Prerequisites

- Python 3.10+ with the ability to create virtual environments
- Node.js 18+ (Vite dev server) and npm
- git (for cloning)

### Backend (FastAPI + pipelines)

1. `cd backend`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install --upgrade pip && pip install -r requirements.txt`
4. Point Pensieve at your library: `export IMAGE_DIR="/absolute/path/to/your/photos"`
5. Pre-download models (optional but avoids first-request lag): `python download_models.py`
6. Build indexes (run once per new library):
   - `python pipeline.py` for image-to-image embeddings (SigLIP → `siglip_index.faiss`)
   - `python clip_search.py` to populate the CLIP text index (`clip_index.faiss`)
7. Start the API: `fastapi run app.py`

### Frontend (React + Vite)

1. `cd frontend`
2. `npm install`
3. `npm run dev` and open the printed URL (default `http://localhost:5173`)

### Keeping the index fresh

- Re-run `python backend/pipeline.py` and the CLIP indexing step whenever you add or delete photos.
- The `/scan` admin route updates metadata and embeddings without wiping existing vectors.
- FAISS index files live alongside the backend (`siglip_index.faiss`, `clip_index.faiss`, `faces*.index`) so back them up with your photos.

## Examples

## Limitations

I really wanted to use mps in this project on my Mac but it causes lots of issues with FAISS
