# BG Remover — AI Background Removal
> FastAPI + rembg · Runs 100% locally · Free · No API keys

---

## Requirements
- Python 3.9 or higher
- pip

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run will download the AI model (~170MB). It's cached automatically after that.

### 2. Start the server
```bash
python app.py
```

### 3. Open your browser
```
http://localhost:8000
```

---

## Features
- ✅ **Batch processing** — upload multiple images at once
- ✅ **Best quality** — uses `u2net_human_seg` model (optimized for people/portraits)
- ✅ **Fast** — model loads once, stays in memory for all requests
- ✅ **GPU support** — automatically uses CUDA if available (`rembg[gpu]`)
- ✅ **Private** — nothing leaves your machine
- ✅ **Download all** — save every result in one click

---

## Changing the AI Model

Edit `app.py` line:
```python
SESSION = new_session("u2net_human_seg")
```

Available models:
| Model | Best For |
|-------|----------|
| `u2net` | General purpose |
| `u2net_human_seg` | Portraits & people ✅ (default) |
| `isnet-general-use` | High detail, general |
| `silueta` | Faster, lighter |
| `u2net_cloth_seg` | Clothing |

---

## Project Structure
```
bgremover/
├── app.py              ← FastAPI backend
├── requirements.txt    ← Python dependencies
├── static/
│   └── index.html      ← Frontend UI
├── uploads/            ← Temp upload storage (auto-created)
└── outputs/            ← Processed PNGs (auto-created)
```

---

## Troubleshooting

**`ModuleNotFoundError: rembg`**
→ Run `pip install -r requirements.txt`

**Port already in use**
→ Change port in `app.py`: `uvicorn.run("app:app", port=8001, ...)`

**Slow on first image**
→ Normal — model is loading into memory. All subsequent images are fast.

**GPU not being used**
→ Install CUDA version: `pip install rembg[gpu]` and ensure CUDA drivers are installed.
