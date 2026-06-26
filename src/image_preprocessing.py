"""
Image preprocessing utilities for the VLM Object Detection Pipeline.
Includes:
  - Resolution scaling and letterbox padding.
  - Contrast enhancement (CLAHE, gamma correction).
  - Denoising (Bilateral, Non-Local Means) and sharpening.
  - Color corrections (sRGB conversion, white balance).
  - Premium transparent grids and Set-of-Mark (SoM) visual prompting.
  - Tiling engine and Non-Maximum Suppression (NMS) for small object merging.
"""

from __future__ import annotations
import math
import numpy as np
import cv2
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("detection_pipeline.preprocessing")

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try a few common truetype fonts, fall back to PIL's default bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()

# ---------------------------------------------------------------------------
# Coordinate Mapping
# ---------------------------------------------------------------------------

def map_bbox_to_original(bbox: list[int], prep_info: dict[str, Any]) -> list[int]:
    """
    Map coordinates [x1, y1, x2, y2] from preprocessed image scale (0-1000)
    back to the original image scale (0-1000).
    Uses floor for min coords and ceil for max coords so bboxes round outward
    and no coverage is lost to rounding at boundaries.
    """
    x1, y1, x2, y2 = bbox
    
    pw = prep_info.get("prep_w", 1000)
    ph = prep_info.get("prep_h", 1000)
    cw = prep_info.get("content_w", pw)
    ch = prep_info.get("content_h", ph)
    pad_l = prep_info.get("pad_left", 0)
    pad_t = prep_info.get("pad_top", 0)
    
    # x_orig_float = (x_prep * pw/1000 - pad_l) * 1000 / cw
    x1_orig = (x1 * pw / 1000 - pad_l) * 1000 / cw
    y1_orig = (y1 * ph / 1000 - pad_t) * 1000 / ch
    x2_orig = (x2 * pw / 1000 - pad_l) * 1000 / cw
    y2_orig = (y2 * ph / 1000 - pad_t) * 1000 / ch
    
    # Round outward: floor for mins, ceil for maxes → no coverage lost at boundaries
    x1_final = max(0, min(1000, math.floor(min(x1_orig, x2_orig))))
    y1_final = max(0, min(1000, math.floor(min(y1_orig, y2_orig))))
    x2_final = max(0, min(1000, math.ceil(max(x1_orig, x2_orig))))
    y2_final = max(0, min(1000, math.ceil(max(y1_orig, y2_orig))))
    
    return [x1_final, y1_final, x2_final, y2_final]

def map_bbox_to_preprocessed(bbox: list[int], prep_info: dict[str, Any]) -> list[int]:
    """
    Map coordinates [x1, y1, x2, y2] from original image scale (0-1000)
    to preprocessed image scale (0-1000).
    """
    x1, y1, x2, y2 = bbox
    
    pw = prep_info.get("prep_w", 1000)
    ph = prep_info.get("prep_h", 1000)
    cw = prep_info.get("content_w", pw)
    ch = prep_info.get("content_h", ph)
    pad_l = prep_info.get("pad_left", 0)
    pad_t = prep_info.get("pad_top", 0)
    
    # px_content = x_orig * content_w / 1000
    # px_prep = px_content + pad_left
    # x_prep = px_prep * 1000 / prep_w
    x1_prep = ((x1 * cw / 1000) + pad_l) * 1000 / pw
    y1_prep = ((y1 * ch / 1000) + pad_t) * 1000 / ph
    x2_prep = ((x2 * cw / 1000) + pad_l) * 1000 / pw
    y2_prep = ((y2 * ch / 1000) + pad_t) * 1000 / ph
    
    x1_final = max(0, min(1000, int(round(min(x1_prep, x2_prep)))))
    y1_final = max(0, min(1000, int(round(min(y1_prep, y2_prep)))))
    x2_final = max(0, min(1000, int(round(max(x1_prep, x2_prep)))))
    y2_final = max(0, min(1000, int(round(max(y1_prep, y2_prep)))))
    
    return [x1_final, y1_final, x2_final, y2_final]

# ---------------------------------------------------------------------------
# Preprocessing Core Functions
# ---------------------------------------------------------------------------

def preprocess_resolution(image: Image.Image, enabled: bool = False, target_short_edge: int = 1024, pad_to_square: bool = False) -> tuple[Image.Image, dict[str, Any]]:
    """Resize and optionally letterbox pad the image to preserve aspect ratio."""
    orig_w, orig_h = image.size
    
    if not enabled:
        return image, {
            "orig_w": orig_w, "orig_h": orig_h,
            "prep_w": orig_w, "prep_h": orig_h,
            "content_w": orig_w, "content_h": orig_h,
            "pad_left": 0, "pad_top": 0
        }
        
    w, h = orig_w, orig_h
    
    # 1. Scale short edge to at least target_short_edge
    short_edge = min(w, h)
    if short_edge < target_short_edge:
        scale = target_short_edge / short_edge
        w, h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((w, h), Image.Resampling.LANCZOS)
        
    content_w, content_h = w, h
    pad_left, pad_top = 0, 0
    prep_w, prep_h = w, h
    
    # 2. Pad to square with neutral gray (128, 128, 128) if requested
    if pad_to_square:
        side = max(w, h)
        pad_left = (side - w) // 2
        pad_top = (side - h) // 2
        padded_image = Image.new("RGB", (side, side), (128, 128, 128))
        padded_image.paste(image, (pad_left, pad_top))
        image = padded_image
        prep_w, prep_h = side, side
        
    prep_info = {
        "orig_w": orig_w, "orig_h": orig_h,
        "prep_w": prep_w, "prep_h": prep_h,
        "content_w": content_w, "content_h": content_h,
        "pad_left": pad_left, "pad_top": pad_top
    }
    return image, prep_info

def preprocess_contrast(image: Image.Image, method: str = "none", clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), gamma: float = 1.0) -> Image.Image:
    """Apply CLAHE or Autocontrast enhancement and optional gamma correction."""
    if method == "none" and gamma == 1.0:
        return image
        
    img_np = np.array(image)
    
    # Apply contrast enhancement
    if method == "clahe":
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_np = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    elif method == "autocontrast":
        # Handled easily by PIL
        image = ImageOps.autocontrast(image)
        img_np = np.array(image)
        
    # Apply Gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        img_np = cv2.LUT(img_np, table)
        
    return Image.fromarray(img_np)

def preprocess_noise_sharpness(image: Image.Image, method: str = "none", sharpen: bool = False) -> Image.Image:
    """Apply bilateral filtering or Non-Local Means filter for noise reduction and unsharp mask."""
    if method == "none" and not sharpen:
        return image
        
    img_np = np.array(image)
    
    # Denoise
    if method == "bilateral":
        img_np = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "nlm":
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
        
    image = Image.fromarray(img_np)
    
    # Sharpen
    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
    return image

def preprocess_color_space(image: Image.Image, white_balance: bool = False) -> Image.Image:
    """Ensure image is sRGB, correct EXIF rotation, and apply Gray World white balance correction."""
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Transpose orientation
    image = ImageOps.exif_transpose(image)
    
    if white_balance:
        img_np = np.array(image).astype(np.float32)
        avg_r = np.mean(img_np[:, :, 0])
        avg_g = np.mean(img_np[:, :, 1])
        avg_b = np.mean(img_np[:, :, 2])
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        
        if avg_r > 0 and avg_g > 0 and avg_b > 0:
            img_np[:, :, 0] = np.clip(img_np[:, :, 0] * (avg_gray / avg_r), 0, 255)
            img_np[:, :, 1] = np.clip(img_np[:, :, 1] * (avg_gray / avg_g), 0, 255)
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * (avg_gray / avg_b), 0, 255)
            
        image = Image.fromarray(img_np.astype(np.uint8))
        
    return image

# ---------------------------------------------------------------------------
# Coordinate Overlays (Advanced Grid & Set-of-Mark)
# ---------------------------------------------------------------------------

def draw_premium_grid(image: Image.Image, style: str = "standard", step: int = 100) -> Image.Image:
    """
    Overlay a premium grid. Supports:
      - 'standard': solid red line grid.
      - 'transparent': semi-transparent blended red grid.
      - 'fine': 10x10 gray coordinate grid (step=100) with thin lines.
      - 'none': no grid overlay.
    """
    if style == "none":
        return image
        
    img = image.copy()
    w, h = img.size
    font = _load_font(max(10, min(w, h) // 70))
    
    if style == "transparent":
        # Draw on transparent overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Grid lines color: semi-transparent red (255, 0, 0, 100)
        grid_color = (255, 0, 0, 100)
        text_color = (255, 255, 255, 255)
        bg_color = (255, 0, 0, 160)
        
        for i in range(0, 1001, step):
            x = i * w / 1000
            draw_overlay.line([(x, 0), (x, h)], fill=grid_color, width=1)
            # Text label
            _text_with_backing(draw_overlay, (x + 2, 2), str(i), font, text_color, bg_color)
            
        for i in range(0, 1001, step):
            y = i * h / 1000
            draw_overlay.line([(0, y), (w, y)], fill=grid_color, width=1)
            _text_with_backing(draw_overlay, (2, y + 2), str(i), font, text_color, bg_color)
            
        # Composite back
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        
    elif style == "fine":
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Thin gray lines
        grid_color = (180, 180, 180, 80)
        text_color = (255, 255, 255, 255)
        bg_color = (70, 70, 70, 180)
        
        for i in range(0, 1001, step):
            x = i * w / 1000
            draw_overlay.line([(x, 0), (x, h)], fill=grid_color, width=1)
            _text_with_backing(draw_overlay, (x + 2, 2), str(i), font, text_color, bg_color)
            
        for i in range(0, 1001, step):
            y = i * h / 1000
            draw_overlay.line([(0, y), (w, y)], fill=grid_color, width=1)
            _text_with_backing(draw_overlay, (2, y + 2), str(i), font, text_color, bg_color)
            
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        
    else:  # standard solid grid
        draw = ImageDraw.Draw(img)
        grid_color = "red"
        for i in range(0, 1001, step):
            x = i * w / 1000
            draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
            _text_with_backing(draw, (x + 2, 2), str(i), font, fill=grid_color)
            
        for i in range(0, 1001, step):
            y = i * h / 1000
            draw.line([(0, y), (w, y)], fill=grid_color, width=1)
            _text_with_backing(draw, (2, y + 2), str(i), font, fill=grid_color)
            
    return img

def _text_with_backing(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont, fill: Any, backing: Any = "black", pad: int = 2):
    """Draw text with a solid backing rectangle."""
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=backing,
    )
    draw.text((x, y), text, fill=fill, font=font)

def generate_som_proposals(image: Image.Image, min_area_pct: float = 0.0005, max_area_pct: float = 0.3) -> tuple[Image.Image, list[dict[str, Any]]]:
    """
    Detect candidate regions using OpenCV contour extraction, overlay colored bounding boxes
    with numbered labels, and return the modified image along with proposal coordinates (0-1000 scale).
    """
    img_np = np.array(image)
    h, w, _ = img_np.shape
    total_area = w * h
    
    # 1. Convert to grayscale and threshold
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to find edge/contour regions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Close small gaps in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 2. Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    candidates = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # Filter out too small or too large regions
        if min_area_pct * total_area < area < max_area_pct * total_area:
            # Map coordinates to 0-1000 scale
            x1 = int(round(x * 1000 / w))
            y1 = int(round(y * 1000 / h))
            x2 = int(round((x + cw) * 1000 / w))
            y2 = int(round((y + ch) * 1000 / h))
            candidates.append([x1, y1, x2, y2])
            
    # Apply a light NMS on proposals to reduce redundant overlaps
    if candidates:
        cleaned_proposals = []
        # Sort by area ascending
        candidates = sorted(candidates, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        while candidates:
            best = candidates.pop(0)
            cleaned_proposals.append(best)
            remaining = []
            for item in candidates:
                # Calculate IoU
                xA = max(best[0], item[0])
                yA = max(best[1], item[1])
                xB = min(best[2], item[2])
                yB = min(best[3], item[3])
                inter = max(0, xB - xA) * max(0, yB - yA)
                area1 = (best[2] - best[0]) * (best[3] - best[1])
                area2 = (item[2] - item[0]) * (item[3] - item[1])
                union = area1 + area2 - inter
                iou_val = inter / union if union > 0 else 0
                if iou_val > 0.4:
                    continue  # suppress
                remaining.append(item)
            candidates = remaining
            
        # Draw proposals on the image with numeric labels
        img_drawn = image.copy()
        draw = ImageDraw.Draw(img_drawn)
        font = _load_font(max(11, min(w, h) // 65))
        
        # Max limit of SoM labels to prevent cluttering (e.g. 30 labels)
        for idx, box in enumerate(cleaned_proposals[:30], 1):
            x1, y1, x2, y2 = box
            
            # Map back to pixels
            px1 = x1 * w / 1000
            py1 = y1 * h / 1000
            px2 = x2 * w / 1000
            py2 = y2 * h / 1000
            
            # Draw semi-transparent cyan boundary
            draw.rectangle([px1, py1, px2, py2], outline="cyan", width=2)
            
            # Label background circle or plate
            label = str(idx)
            _text_with_backing(draw, (px1 + 2, py1 + 2), f"#{label}", font, fill="white", backing="cyan", pad=2)
            
            proposals.append({
                "id": idx,
                "bbox_2d": box
            })
        return img_drawn, proposals
        
    return image, []

# ---------------------------------------------------------------------------
# Tiling Engine
# ---------------------------------------------------------------------------

def get_image_tiles(image: Image.Image, tile_size: int = 512, overlap_pct: float = 0.2) -> list[dict[str, Any]]:
    """Divide the input image into overlapping tiles."""
    W, H = image.size
    step = int(round(tile_size * (1.0 - overlap_pct)))
    if step <= 0:
        step = tile_size
        
    tiles = []
    
    y_coords = list(range(0, H - tile_size + 1, step))
    if not y_coords or y_coords[-1] + tile_size < H:
        y_coords.append(max(0, H - tile_size))
        
    x_coords = list(range(0, W - tile_size + 1, step))
    if not x_coords or x_coords[-1] + tile_size < W:
        x_coords.append(max(0, W - tile_size))
        
    # De-duplicate coords
    y_coords = sorted(list(set(y_coords)))
    x_coords = sorted(list(set(x_coords)))
    
    for y in y_coords:
        for x in x_coords:
            tw = min(tile_size, W - x)
            th = min(tile_size, H - y)
            
            tile_img = image.crop((x, y, x + tw, y + th))
            tiles.append({
                "tile_image": tile_img,
                "tile_x": x,
                "tile_y": y,
                "tile_w": tw,
                "tile_h": th
            })
            
    return tiles

def map_tile_detection_to_original(bbox: list[int], tile_x: int, tile_y: int, tile_w: int, tile_h: int, orig_w: int, orig_h: int) -> list[int]:
    """Map coordinate bounding box from a local tile scale (0-1000) to original image scale (0-1000)."""
    tx1, ty1, tx2, ty2 = bbox
    
    # Local tile pixel coordinates
    px1_tile = tx1 * tile_w / 1000
    py1_tile = ty1 * tile_h / 1000
    px2_tile = tx2 * tile_w / 1000
    py2_tile = ty2 * tile_h / 1000
    
    # Original image pixel coordinates
    px1_orig = tile_x + px1_tile
    py1_orig = tile_y + py1_tile
    px2_orig = tile_x + px2_tile
    py2_orig = tile_y + py2_tile
    
    # Original image 0-1000 coordinates
    x1_orig = px1_orig * 1000 / orig_w
    y1_orig = py1_orig * 1000 / orig_h
    x2_orig = px2_orig * 1000 / orig_w
    y2_orig = py2_orig * 1000 / orig_h
    
    x1_final = max(0, min(1000, int(round(min(x1_orig, x2_orig)))))
    y1_final = max(0, min(1000, int(round(min(y1_orig, y2_orig)))))
    x2_final = max(0, min(1000, int(round(max(x1_orig, x2_orig)))))
    y2_final = max(0, min(1000, int(round(max(y1_orig, y2_orig)))))
    
    return [x1_final, y1_final, x2_final, y2_final]

# ---------------------------------------------------------------------------
# Non-Maximum Suppression (NMS)
# ---------------------------------------------------------------------------

def calculate_iou(box1: list[int], box2: list[int]) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xA = max(x1_1, x1_2)
    yA = max(y1_1, y1_2)
    xB = min(x2_1, x2_2)
    yB = min(y2_1, y2_2)
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union

def apply_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to filter out overlapping duplicate bounding boxes.
    Sorts by bounding box area ascending to favor tighter detections.
    """
    if not detections:
        return []
        
    # Sort detections by area of bounding box ascending (smaller/tighter boxes first)
    def box_area(d):
        b = d["bbox_2d"]
        return (b[2] - b[0]) * (b[3] - b[1])
        
    sorted_dets = sorted(detections, key=box_area)
    keep = []
    
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        remaining = []
        for det in sorted_dets:
            if det["label"] == best["label"]:
                iou_val = calculate_iou(best["bbox_2d"], det["bbox_2d"])
                if iou_val > iou_threshold:
                    continue  # Suppress duplicate detection
            remaining.append(det)
        sorted_dets = remaining
        
    return keep
