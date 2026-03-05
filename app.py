"""
BG Remover — FastAPI + rembg backend
Run:  pip install -r requirements.txt
      python app.py
Then open:  http://localhost:8000
"""

import io
import uuid
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rembg import remove, new_session

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="BG Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = Path("uploads")
OUTPUT_DIR  = Path("outputs")
STATIC_DIR  = Path("static")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Pre-load the model once at startup for fast responses
# Models available: u2net (default), u2net_human_seg (best for people), isnet-general-use
print("⏳ Loading AI model (u2net_human_seg) — first run downloads ~170MB …")
SESSION = new_session("u2net_human_seg")
print("✅ Model ready!")

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main UI."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found. Make sure static/index.html exists.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """
    Accept an image upload, remove its background, return the transparent PNG.
    """
    # Validate
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > 15 * 1024 * 1024:  # 15 MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 15MB.")

    try:
        # Open with PIL
        input_image = Image.open(io.BytesIO(file_bytes)).convert("RGBA")

        # Run background removal
        output_image = remove(input_image, session=SESSION)

        # Save output
        output_filename = f"{uuid.uuid4().hex}.png"
        output_path = OUTPUT_DIR / output_filename
        output_image.save(output_path, format="PNG")

        return JSONResponse({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "width": output_image.width,
            "height": output_image.height,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{filename}")
async def download_result(filename: str):
    """Download a processed image by filename."""
    # Sanitize — no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired.")

    original_name = filename  # could be customized
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=f"removed-bg-{filename}",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "u2net_human_seg"}


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting BG Remover server at http://localhost:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
