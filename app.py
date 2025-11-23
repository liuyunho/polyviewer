#!/usr/bin/env python

from flask import Flask, request, send_file, Response
import io
from pathlib import Path

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

    # Read full content into memory
    data = file_storage.read()
    bio = io.BytesIO(data)  # make a file-like object at position 0

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

    # (C, H, W)
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] == arr.shape[2]:
        c = arr.shape[0]
        if c == 1:
            return arr[0]
        # Use first 3 channels as RGB
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

    # Fallback: just take first slice
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
  <title>Python-backed Paired Image Viewer</title>
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
      gap: 16px;
      align-items: center;
      font-size: 14px;
    }
    .controls input[type="file"] {
      font-size: 12px;
    }
    #viewerContainer {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
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
    }
  </style>
</head>
<body>
  <div class="controls">
    <label>
      Left image:
      <input id="fileLeft" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp">
    </label>
    <label>
      Right image:
      <input id="fileRight" type="file" accept=".tif,.tiff,.ome.tif,.ome.tiff,.png,.jpg,.jpeg,.webp">
    </label>
    <span class="status" id="status">Choose two images (Python will process on upload)</span>
  </div>

  <div id="viewerContainer">
    <canvas id="canvasLeft"></canvas>
    <canvas id="canvasRight"></canvas>
  </div>

  <script>
    const fileLeft = document.getElementById('fileLeft');
    const fileRight = document.getElementById('fileRight');
    const canvasLeft = document.getElementById('canvasLeft');
    const canvasRight = document.getElementById('canvasRight');
    const statusEl = document.getElementById('status');

    let imgLeft = null;
    let imgRight = null;
    let imgWidth = null;
    let imgHeight = null;

    let scale = 1.0;
    let offsetX = 0;
    let offsetY = 0;

    let isPanning = false;
    let lastX = 0;
    let lastY = 0;

    function setStatus(msg) {
      statusEl.textContent = msg;
    }

    function setLoadedImage(img, side) {
      const w = img.width;
      const h = img.height;

      if (!imgWidth || !imgHeight) {
        imgWidth = w;
        imgHeight = h;
        setStatus(`Image size: ${imgWidth} × ${imgHeight}px`);
      } else if (w !== imgWidth || h !== imgHeight) {
        alert(
          `The two images must have the same size.\n` +
          `First: ${imgWidth}×${imgHeight}, new: ${w}×${h}`
        );
        return;
      }

      if (side === 'left') {
        imgLeft = img;
      } else {
        imgRight = img;
      }

      resetView();
      draw();
    }

    async function uploadAndLoad(file, side) {
      const formData = new FormData();
      formData.append("file", file);

      setStatus(`Uploading ${file.name} to Python backend...`);

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

    fileLeft.addEventListener('change', e => {
      if (e.target.files && e.target.files[0]) {
        uploadAndLoad(e.target.files[0], 'left');
      }
    });

    fileRight.addEventListener('change', e => {
      if (e.target.files && e.target.files[0]) {
        uploadAndLoad(e.target.files[0], 'right');
      }
    });

    function resizeCanvases() {
      const rectL = canvasLeft.getBoundingClientRect();
      const rectR = canvasRight.getBoundingClientRect();

      if (canvasLeft.width !== rectL.width || canvasLeft.height !== rectL.height) {
        canvasLeft.width = rectL.width;
        canvasLeft.height = rectL.height;
      }
      if (canvasRight.width !== rectR.width || canvasRight.height !== rectR.height) {
        canvasRight.width = rectR.width;
        canvasRight.height = rectR.height;
      }
    }

    function resetView() {
      if (!imgWidth || !imgHeight) return;

      resizeCanvases();
      const rect = canvasLeft.getBoundingClientRect();

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
      const ctxL = canvasLeft.getContext('2d');
      const ctxR = canvasRight.getContext('2d');

      drawOne(ctxL, canvasLeft, imgLeft);
      drawOne(ctxR, canvasRight, imgRight);

      if (imgWidth && imgHeight) {
        setStatus(`Zoom: ${(scale * 100).toFixed(1)}%`);
      }
    }

    function getCanvasCoords(canvas, event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
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

    function handleMouseDown(canvas, event) {
      if (!imgWidth || !imgHeight) return;
      isPanning = true;
      const { x, y } = getCanvasCoords(canvas, event);
      lastX = x;
      lastY = y;
    }

    function handleMouseMove(canvas, event) {
      if (!isPanning) return;
      const { x, y } = getCanvasCoords(canvas, event);
      const dx = x - lastX;
      const dy = y - lastY;
      lastX = x;
      lastY = y;

      offsetX += dx;
      offsetY += dy;

      draw();
    }

    function handleMouseUp() {
      isPanning = false;
    }

    function attachInteractions(canvas) {
      canvas.addEventListener('wheel', e => handleWheel(canvas, e), { passive: false });
      canvas.addEventListener('mousedown', e => handleMouseDown(canvas, e));
      canvas.addEventListener('mousemove', e => handleMouseMove(canvas, e));
      window.addEventListener('mouseup', handleMouseUp);
      canvas.addEventListener('mouseleave', handleMouseUp);
    }

    attachInteractions(canvasLeft);
    attachInteractions(canvasRight);

    window.addEventListener('resize', () => {
      if (imgWidth && imgHeight) {
        draw();
      } else {
        resizeCanvases();
      }
    });

    resizeCanvases();
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
