"""
BG Remover — FastAPI + rembg backend
Supports: Portraits, Dark images, Signatures/Documents
Model: isnet-general-use (best general purpose)
"""

import io
import uuid
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from rembg import remove, new_session

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="BG Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Single general-purpose model — good for people, objects, dark images
print("⏳ Loading AI model (isnet-general-use) — first run downloads ~170MB …")
SESSION = new_session("isnet-general-use")
print("✅ Model ready!")


# ── Image Analysis Helpers ────────────────────────────────────────────────────

def get_image_stats(img_gray: Image.Image) -> dict:
    """Analyze brightness, contrast, and whether it looks like a document."""
    arr = np.array(img_gray, dtype=np.float32)
    mean_brightness = float(arr.mean())
    std_contrast    = float(arr.std())

    # Count very light pixels (>220) — high ratio = document/signature
    light_pixel_ratio = float((arr > 220).sum() / arr.size)

    # Count very dark pixels (<30) — high ratio = dark background
    dark_pixel_ratio  = float((arr < 30).sum() / arr.size)

    return {
        "brightness":        mean_brightness,
        "contrast":          std_contrast,
        "light_pixel_ratio": light_pixel_ratio,
        "dark_pixel_ratio":  dark_pixel_ratio,
        "is_document":       light_pixel_ratio > 0.55 and std_contrast < 80,
        "is_dark":           mean_brightness < 60 or dark_pixel_ratio > 0.50,
        "is_low_contrast":   std_contrast < 35,
    }


def preprocess_dark_image(img: Image.Image) -> Image.Image:
    """Boost brightness + contrast for dark/low-contrast images."""
    img = ImageEnhance.Brightness(img).enhance(2.2)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


def remove_signature_background(img: Image.Image) -> Image.Image:
    """
    Threshold-based removal for signatures and documents.
    Dark ink on light background — far better than AI for this case.
    """
    gray = img.convert("L")
    arr  = np.array(gray, dtype=np.uint8)

    # Use 75th percentile brightness as ink/background threshold
    threshold = int(np.percentile(arr, 75))
    threshold = min(threshold, 200)

    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.uint8)

    # Dark pixels = ink = opaque; Light pixels = background = transparent
    alpha = np.where(arr < threshold, 255, 0).astype(np.uint8)

    # Soften edges
    alpha_img = Image.fromarray(alpha, mode='L')
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    data[:, :, 3] = np.array(alpha_img)

    return Image.fromarray(data, 'RGBA')


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    """
    Smart background removal:
    - Signatures/documents  → threshold-based (pixel-perfect ink extraction)
    - Dark/low-contrast     → AI + brightness/contrast pre-processing
    - Normal images         → AI with isnet-general-use model
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 15MB.")

    try:
        original = Image.open(io.BytesIO(file_bytes)).convert("RGBA")
        gray     = original.convert("L")
        stats    = get_image_stats(gray)

        print(f"  Image: {file.filename} | {original.width}x{original.height}")
        print(f"  Stats: brightness={stats['brightness']:.1f}, contrast={stats['contrast']:.1f}, "
              f"light={stats['light_pixel_ratio']:.2f}, dark={stats['dark_pixel_ratio']:.2f}")

        if stats["is_document"]:
            # Signature / Document mode
            print("  → Document/signature detected — using threshold mode")
            output_image = remove_signature_background(original)
            method = "document"

        elif stats["is_dark"] or stats["is_low_contrast"]:
            # Dark image — pre-process then AI
            print(f"  → Dark image detected — enhancing then running AI")
            processed = preprocess_dark_image(original)
            ai_result = remove(processed, session=SESSION)

            # Apply AI alpha onto original colors (not the brightened version)
            ai_arr   = np.array(ai_result)
            orig_arr = np.array(original)
            orig_arr[:, :, 3] = ai_arr[:, :, 3]
            output_image = Image.fromarray(orig_arr, 'RGBA')
            method = "ai-enhanced"

        else:
            # Normal AI mode
            print("  → Normal image — running AI")
            output_image = remove(original, session=SESSION)
            method = "ai"

        # Save result
        output_filename = f"{uuid.uuid4().hex}.png"
        output_path     = OUTPUT_DIR / output_filename
        output_image.save(output_path, format="PNG")

        print(f"  Done — method={method}")

        return JSONResponse({
            "success":      True,
            "filename":     output_filename,
            "download_url": f"/download/{output_filename}",
            "width":        output_image.width,
            "height":       output_image.height,
            "method":       method,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/download/{filename}")
async def download_result(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found or expired.")
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=f"removed-bg-{filename}",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "isnet-general-use", "version": "2.0"}


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting BG Remover at http://localhost:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)