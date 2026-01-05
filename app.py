#!/usr/bin/env python

from flask import Flask, request, send_file, Response
import io

import numpy as np
import tifffile as tiff
from PIL import Image

app = Flask(__name__)

# ---------- Python helpers (runs when you upload a file) ----------

def load_tif_or_image(file_storage):
    """
    Read TIFF/OME-TIFF/PNG/JPEG into a numpy array using tifffile/Pillow.
    We first read the uploaded file into memory (bytes) and wrap it in BytesIO,
    so tifffile/Pillow always see a clean file-like object.
    """
    filename = file_storage.filename or ""
    name_lower = filename.lower()

    data = file_storage.read()
    bio = io.BytesIO(data)

    # TIFF / OME-TIFF via tifffile
    if name_lower.endswith(".tif") or name_lower.endswith(".tiff") \
       or name_lower.endswith(".ome.tif") or name_lower.endswith(".ome.tiff"):
        arr = tiff.imread(bio)
        print(f"Loaded TIFF {filename} with shape {arr.shape}, dtype {arr.dtype}")
        return arr

    # Other images via Pillow (PNG, JPEG, etc.)
    img = Image.open(bio)
    arr = np.array(img)
    print(f"Loaded image {filename} with shape {arr.shape}, dtype {arr.dtype}")
    return arr


def select_display_array(arr):
    """
    Convert arbitrary shapes to something displayable:
      - (H, W)           -> grayscale
      - (C, H, W)        -> channels-first (pick first or RGB)
      - (H, W, C)        -> channels-last (pick first or RGB)
      - (Z, C, H, W)     -> pick Z=0, C=0
      - (Z, H, W)        -> pick Z=0
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    # OME-like: (Z, C, H, W)
    if arr.ndim == 4:
        z, c, h, w = arr.shape
        print(f"Treating as (Z,C,H,W), picking Z=0,C=0 from shape {arr.shape}")
        return arr[0, 0]

    # (C, H, W) heuristic
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] == arr.shape[2]:
        c = arr.shape[0]
        if c == 1:
            return arr[0]
        print(f"Treating as (C,H,W), using first 3 of C={c}")
        return np.moveaxis(arr[:3, :, :], 0, -1)

    # (H, W, C)
    if arr.ndim == 3 and arr.shape[-1] <= 16:
        c = arr.shape[-1]
        if c <= 4:
            print(f"Treating as (H,W,C) with C={c}")
            return arr
        print(f"(H,W,C) with C={c}, using first 3 channels as RGB")
        return arr[..., :3]

    print(f"Unknown shape {arr.shape}, using first slice")
    return np.squeeze(arr[0])


def normalize_to_uint8(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype("float32")
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype("uint8")


def array_to_png_bytes(arr):
    arr = normalize_to_uint8(arr)
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mode = "RGBA"
    else:
        print("Unexpected shape for PNG, converting to grayscale")
        arr = np.mean(arr, axis=-1)
        arr = normalize_to_uint8(arr)
        mode = "L"

    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---------- Flask routes ----------

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Python-backed 4-Panel Synced Image Viewer</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    .controls {
      padding: 8px 12px;
      border-bottom: 1px solid #ddd;
      display: flex;
      gap: 12px;
      align-items: center;
      font-size: 14px;
      flex-wrap: wrap;
    }
    .controls input[type="file"] {
      font-size: 12px;
    }
    .btn {
      padding: 6px 10px;
      border: 1px solid #ccc;
      border-radius: 8px;
      background: #fafafa;
      cursor: pointer;
      font-size: 12px;
    }
    .btn:hover { background: #f2f2f2; }

    #viewerContainer {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      gap: 1px;
      background: #ccc;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      background: #111;
      cursor: grab;
    }
    canvas:active {
      cursor: grabbing;
    }
    .status {
      margin-left: auto;
      font-size: 12px;
      color: #555;
      white-space: nowrap;
    }
    .hint {
      font-size: 12px;
      color: #666;
    }
  </style>
</head>
<body>
  <div class="controls">
    <label>TL: <input id="file_tl" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp"></label>
    <label>TR: <input id="file_tr" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp"></label>
    <label>BL: <input id="file_bl" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp"></label>
    <label>BR: <input id="file_br" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp"></label>

    <button class="btn" id="btnReset">Reset view</button>
    <span class="hint">Wheel=zoom, Drag=pan (synced)</span>

    <span class="status" id="status">Choose up to 4 images (Python will process on upload)</span>
  </div>

  <div id="viewerContainer">
    <canvas id="canvas_tl"></canvas>
    <canvas id="canvas_tr"></canvas>
    <canvas id="canvas_bl"></canvas>
    <canvas id="canvas_br"></canvas>
  </div>

  <script>
    const statusEl = document.getElementById('status');
    const btnReset = document.getElementById('btnReset');

    const sides = ['tl', 'tr', 'bl', 'br'];

    const fileInputs = {
      tl: document.getElementById('file_tl'),
      tr: document.getElementById('file_tr'),
      bl: document.getElementById('file_bl'),
      br: document.getElementById('file_br'),
    };

    const canvases = {
      tl: document.getElementById('canvas_tl'),
      tr: document.getElementById('canvas_tr'),
      bl: document.getElementById('canvas_bl'),
      br: document.getElementById('canvas_br'),
    };

    const images = { tl: null, tr: null, bl: null, br: null };

    let imgWidth = null;
    let imgHeight = null;

    // Shared view state (synced across all panels)
    let scale = 1.0;
    let offsetX = 0;
    let offsetY = 0;

    // Pan state (use window-level mousemove to avoid cross-canvas jumps)
    let isPanning = false;
    let lastClientX = 0;
    let lastClientY = 0;

    function setStatus(msg) { statusEl.textContent = msg; }

    function resizeCanvases() {
      for (const s of sides) {
        const c = canvases[s];
        const rect = c.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        if (c.width !== w) c.width = w;
        if (c.height !== h) c.height = h;
      }
    }

    function resetView() {
      if (!imgWidth || !imgHeight) return;

      resizeCanvases();
      const rect = canvases.tl.getBoundingClientRect();

      const scaleX = rect.width / imgWidth;
      const scaleY = rect.height / imgHeight;
      scale = Math.min(scaleX, scaleY);

      offsetX = (rect.width - imgWidth * scale) / 2;
      offsetY = (rect.height - imgHeight * scale) / 2;
    }

    function drawOne(ctx, canvas, img) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!img) return;

      ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
      ctx.drawImage(img, 0, 0);
    }

    function draw() {
      resizeCanvases();

      for (const s of sides) {
        const c = canvases[s];
        const ctx = c.getContext('2d');
        drawOne(ctx, c, images[s]);
      }

      if (imgWidth && imgHeight) {
        setStatus(`Image size: ${imgWidth}×${imgHeight}px  |  Zoom: ${(scale * 100).toFixed(1)}%`);
      } else {
        setStatus('Choose up to 4 images (Python will process on upload)');
      }
    }

    function setLoadedImage(img, side) {
      const w = img.width;
      const h = img.height;

      if (!imgWidth || !imgHeight) {
        imgWidth = w;
        imgHeight = h;
      } else if (w !== imgWidth || h !== imgHeight) {
        alert(`All images must have the same size.\nExpected: ${imgWidth}×${imgHeight}\nGot: ${w}×${h}`);
        return;
      }

      images[side] = img;

      // reset view whenever a new image comes in (same behavior as your old version)
      resetView();
      draw();
    }

    async function uploadAndLoad(file, side) {
      const formData = new FormData();
      formData.append("file", file);

      setStatus(`Uploading (${side}) ${file.name} ...`);

      const resp = await fetch(`/api/upload/${side}`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const msg = await resp.text();
        alert("Upload failed: " + msg);
        setStatus("Upload failed");
        return;
      }

      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        setLoadedImage(img, side);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    }

    // Bind file inputs
    for (const s of sides) {
      fileInputs[s].addEventListener('change', e => {
        const f = e.target.files && e.target.files[0];
        if (f) uploadAndLoad(f, s);
      });
    }

    // Zoom helper
    function getCanvasCoords(canvas, event) {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function handleWheel(canvas, event) {
      if (!imgWidth || !imgHeight) return;
      event.preventDefault();

      const { x, y } = getCanvasCoords(canvas, event);
      const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;
      const newScale = scale * zoomFactor;

      const minScale = 0.05;
      const maxScale = 20;
      if (newScale < minScale || newScale > maxScale) return;

      const worldX = (x - offsetX) / scale;
      const worldY = (y - offsetY) / scale;

      scale = newScale;
      offsetX = x - worldX * scale;
      offsetY = y - worldY * scale;

      draw();
    }

    function handleMouseDown(event) {
      if (!imgWidth || !imgHeight) return;
      isPanning = true;
      lastClientX = event.clientX;
      lastClientY = event.clientY;
    }

    function handleMouseMove(event) {
      if (!isPanning) return;
      const dx = event.clientX - lastClientX;
      const dy = event.clientY - lastClientY;
      lastClientX = event.clientX;
      lastClientY = event.clientY;

      offsetX += dx;
      offsetY += dy;
      draw();
    }

    function handleMouseUp() { isPanning = false; }

    // Attach interactions to all canvases (wheel + mousedown)
    for (const s of sides) {
      const c = canvases[s];
      c.addEventListener('wheel', e => handleWheel(c, e), { passive: false });
      c.addEventListener('mousedown', handleMouseDown);
    }
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Reset button
    btnReset.addEventListener('click', () => {
      resetView();
      draw();
    });

    // Resize
    window.addEventListener('resize', () => {
      if (imgWidth && imgHeight) resetView();
      draw();
    });

    resizeCanvases();
    draw();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/api/upload/<side>")
def upload(side):
    """
    Receive uploaded file, run tiff.imread / Pillow in Python,
    convert to PNG, return PNG bytes.
    side can be: tl/tr/bl/br (or any string; we don't persist it server-side).
    """
    if "file" not in request.files:
        return "No file uploaded", 400

    file_storage = request.files["file"]
    if not file_storage.filename:
        return "No filename", 400

    try:
        arr = load_tif_or_image(file_storage)
        arr_disp = select_display_array(arr)
        png_buf = array_to_png_bytes(arr_disp)
    except Exception as e:
        print("Error processing image:", e)
        return f"Error: {e}", 500

    return send_file(
        png_buf,
        mimetype="image/png",
        as_attachment=False,
        download_name="image.png",
    )


if __name__ == "__main__":
    # Access at http://127.0.0.1:5000
    app.run(debug=True)
