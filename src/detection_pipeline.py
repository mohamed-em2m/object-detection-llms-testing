"""
Object detection pipeline: a VLM "detector" agent proposes bounding
boxes for objects in an image, a VLM "judge" agent critiques them against
the original image, and the loop repeats with feedback until a score
threshold is hit or rounds run out.

Key features:
  - Robust JSON parsing: handles <answer> blocks also wrapped in code fences.
  - Detection validation: drops/clamps malformed bboxes and unknown labels.
  - Readable overlays: real fonts with background plates for grid numbers
    and box labels.
  - Retry/backoff around every API call.
  - max_tokens set explicitly on the detector call.
  - Logging with per-round summaries.
  - Persistence: best annotated image, detections JSON, and full round
    history are written to disk.
  - Prompts are loaded from src/prompts/*.md files (with hardcoded fallbacks).
  - Basic input validation (image exists, categories non-empty).
"""

from __future__ import annotations
import os
os.environ['MPLBACKEND'] = 'Agg'
import json
import logging
import re
import time
import traceback
import io
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from openai import OpenAI
from json_repair import repair_json

from image_preprocessing import (
    preprocess_resolution,
    preprocess_contrast,
    preprocess_noise_sharpness,
    preprocess_color_space,
    draw_premium_grid,
    generate_som_proposals,
    get_image_tiles,
    map_tile_detection_to_original,
    map_bbox_to_original,
    map_bbox_to_preprocessed,
    apply_nms,
)

logger = logging.getLogger("detection_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Prompt Templates — loaded from prompts/ directory, with fallbacks
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_template(filename: str, fallback: str) -> str:
    path = PROMPTS_DIR / filename
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to load prompt from %s, using fallback: %s", path, exc)
    return fallback


DEFAULT_DETECTOR_FALLBACK = """
You are a meticulous annotation assistant performing object detection.

## Categories to detect
{categories_list}

## Category definitions (use these to disambiguate visually similar categories)
{category_definitions}
{feedback_block}

## Task
Analyze the image and detect every visible instance of the categories above. Work through the following steps internally before producing your final answer:

1. Systematic scan: Mentally divide the image into a grid (e.g. top-left, top-right, center, bottom-left, bottom-right, and any remaining regions) and inspect each region in turn for target categories.
2. Candidate identification: For each candidate object found, note its approximate location and visual characteristics (shape, color, boundaries, texture).
3. Classification: Match each candidate against the category definitions above. If a candidate could fit two categories, use the distinguishing details to pick the single best label. Discard candidates that don't clearly match any category.
4. Bounding box estimation: Using the image's grid and axis labels as reference, estimate a TIGHT bounding box around each confirmed object on a 0-1000 scale, where (0,0) is top-left and (1000,1000) is bottom-right. The box should hug the visible extent of the target, not surrounding background.
5. Deduplication check: Verify no single object is reported twice with overlapping/near-identical boxes, and verify no region was skipped.
6. Final compilation: List only the objects you are confident are genuinely present and visible. If none are found for a category, omit it entirely. If no targets are visible at all, the final array should be empty.

## Output format
Respond in exactly two parts, in this order:

<answer>
[
  {{
    "label": "category_name",
    "bbox_2d": [x1, y1, x2, y2]
  }}
]
</answer>

## Rules
- Coordinates must be integers on a 0-1000 scale, with x1 < x2 and y1 < y2.
- "label" must be exactly one of: {categories_list}.
- The content inside <answer> must be ONLY valid JSON (a JSON array, possibly empty: []) — no comments, no trailing commas, no extra text, and NOT wrapped in code fences.
- Do not invent or guess at objects that are not clearly visible; when uncertain, exclude the candidate.
- Do not include the <analysis> reasoning inside the <answer> block.
"""

DEFAULT_JUDGE_FALLBACK = """
You are a strict quality auditor for object detection annotations.

You are shown two images of the same subject, both with a red coordinate grid (0-1000 scale,
(0,0) top-left, (1000,1000) bottom-right):
1. The ORIGINAL image (no boxes drawn) — use this to judge what target objects actually exist.
2. The ANNOTATED image, where a detection agent has drawn lime-green bounding boxes with labels.

## Categories and definitions
{category_definitions}

## Your job
Critically compare the two images and evaluate the annotated image's quality:
1. Coverage: are there visible target objects in the original image that were NOT detected? List each with its approximate (x,y) grid location.
2. Correctness: for each detected box, is the label correct given the definitions above? List any mislabeled boxes and what they should be instead.
3. False positives: any boxes drawn over background with no real target object? List them.
4. Bounding box quality: for each box, is it tight around the object, or too loose / too tight / offset? Give specific fixes referencing approximate coordinates.
5. Duplicates: any single object annotated more than once with overlapping boxes?

## Output
Respond in exactly this format, nothing else:

<score>N</score>
<feedback>
A concise, actionable bullet list of concrete fixes for the next detection attempt, each with
approximate 0-1000 coordinates where relevant (e.g. "Missed a small target near (650,300)",
"Box labeled 'A' near (200,800) should be 'B'", "Tighten the box at top-left, left edge extends ~40px into empty area", "Remove duplicate box near (400,400)").
If the annotation is already excellent, state that explicitly and say no changes are needed.
</feedback>

N is an integer 0-10 (10 = perfect coverage, correct labels, tight boxes, no false positives or duplicates).

Raw JSON detections produced by the agent, for reference:
{detections_json}
"""

DEFAULT_DETECTOR_TEMPLATE = _load_prompt_template("detector_agent.md", DEFAULT_DETECTOR_FALLBACK)
DEFAULT_JUDGE_TEMPLATE = _load_prompt_template("feedback_agent.md", DEFAULT_JUDGE_FALLBACK)


# ---------------------------------------------------------------------------
# Image / font helpers
# ---------------------------------------------------------------------------

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


def _text_with_backing(draw: ImageDraw.ImageDraw, xy, text, font, fill, backing="black", pad=2):
    """Draw text with a solid backing rectangle so it stays legible over photos."""
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=backing,
    )
    draw.text((x, y), text, fill=fill, font=font)


def draw_grid(image: Image.Image, step: int = 100) -> Image.Image:
    """Overlay a 0-1000 scale coordinate grid with readable axis labels."""
    return draw_premium_grid(image, style="standard", step=step)


def render_detections(base_image: Image.Image, detections: list[dict]) -> Image.Image:
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(max(12, min(w, h) // 50))

    for item in detections:
        bbox = item.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        left = xmin * w / 1000
        top = ymin * h / 1000
        right = xmax * w / 1000
        bottom = ymax * h / 1000
        draw.rectangle([left, top, right, bottom], outline="lime", width=4)
        label_y = max(0, top - 18)
        _text_with_backing(draw, (left + 2, label_y), item.get("label", "object"), font, fill="lime")
    return img


def pil_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{encoded}"


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking-mode models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Recursively strip markdown code fences (``` … ```) from text."""
    text = text.strip()
    # Handle fences that may have a language tag on the opening line
    changed = True
    while changed:
        new = re.sub(r"^```[a-zA-Z]*\r?\n?(.*?)```\s*$", r"\1", text, flags=re.DOTALL)
        new = new.strip()
        changed = new != text
        text = new
    return text


def _extract_balanced_array(text: str) -> str:
    """
    Find the outermost JSON array in *text* by scanning for balanced brackets.
    Returns the matched substring, or *text* unchanged if no array is found.
    This is more robust than ``text[find('['):rfind(']')+1]`` because it
    won't be tricked by judge-feedback text that also contains ``[…]`` spans.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
            if depth == 0 and start is not None:
                return text[start: i + 1]
    return text


def extract_json_block(text: str) -> str:
    """Best-effort extraction of a JSON array from free-form model text."""
    text = _strip_code_fences(text)
    if "[" in text:
        return _extract_balanced_array(text)
    return text


def parse_detections(raw_text: str) -> list[dict]:
    """
    Parse the model's raw response into a list of detection dicts.
    Raises ValueError (with the offending text attached) on failure so callers
    can log/inspect it instead of silently losing the round's output.
    """
    # 1. Strip thinking blocks so they don't confuse the extractors
    cleaned = _strip_think_blocks(raw_text)

    # 2. Prefer the content inside <answer>…</answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", cleaned, re.DOTALL)
    candidate = answer_match.group(1).strip() if answer_match else cleaned

    # 3. Strip any remaining code fences and extract a balanced JSON array
    json_block = extract_json_block(candidate)

    try:
        repaired = repair_json(json_block)
        parsed = json.loads(repaired)
    except Exception as exc:
        raise ValueError(
            f"Could not parse detections JSON: {exc}\nRaw text was:\n{raw_text}"
        ) from exc

    # Some models wrap the list in a dict, e.g. {"detections": [...]}
    if isinstance(parsed, dict):
        for key in ("detections", "objects", "results", "items", "data"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            # Last resort: take the first list-valued field
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array of detections, got: {type(parsed).__name__}")
    return parsed


def validate_detections(detections: list[dict], categories: list[str]) -> list[dict]:
    """
    Drop malformed entries (bad label, bad/degenerate bbox) instead of letting
    them silently corrupt rendering and the judge prompt. Logs what it drops.
    """
    valid_labels = set(categories)
    cleaned = []
    for i, item in enumerate(detections):
        if not isinstance(item, dict):
            logger.warning("Dropping detection #%d: not an object (%r)", i, item)
            continue

        label = item.get("label")
        if label not in valid_labels:
            logger.warning("Dropping detection #%d: unknown label %r", i, label)
            continue

        bbox = item.get("bbox_2d")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            logger.warning("Dropping detection #%d (%s): malformed bbox %r", i, label, bbox)
            continue

        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            logger.warning("Dropping detection #%d (%s): non-numeric bbox %r", i, label, bbox)
            continue

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = max(0, min(1000, x1)), max(0, min(1000, x2))
        y1, y2 = max(0, min(1000, y1)), max(0, min(1000, y2))

        if x2 - x1 < 1 or y2 - y1 < 1:
            logger.warning("Dropping detection #%d (%s): degenerate bbox after clamping %r", i, label, bbox)
            continue

        cleaned.append({"label": label, "bbox_2d": [int(x1), int(y1), int(x2), int(y2)]})
    return cleaned


# ---------------------------------------------------------------------------
# Retry helper for API calls
# ---------------------------------------------------------------------------

def _call_with_retries(fn, *, retries: int = 3, base_delay: float = 1.5, what: str = "API call"):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("%s failed (attempt %d/%d): %s", what, attempt, retries, exc)
            if attempt < retries:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"{what} failed after {retries} attempts") from last_exc


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round: int
    detections: list
    score: int
    feedback: str
    raw_detector_output: str
    parse_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Object Detection Pipeline
# ---------------------------------------------------------------------------

class ObjectDetectionPipeline:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        detector_client: Optional[OpenAI] = None,
        judge_client: Optional[OpenAI] = None,
        detector_model: str = "gpt-4.1",
        judge_model: str = "gpt-4.1",
        max_rounds: int = 1,
        score_threshold: int = 8,
        detector_template: str = DEFAULT_DETECTOR_TEMPLATE,
        judge_template: str = DEFAULT_JUDGE_TEMPLATE,
        detector_max_tokens: int = 4096,
        judge_max_tokens: int = 1024,
        api_retries: int = 3,
        detector_temperature: float = 0.9,
        detector_top_p: float = 0.95,
        judge_temperature: float = 0.2,
        preprocessing_config: Optional[dict] = None,
    ):
        """
        `client` is used for both detector and judge calls unless overridden by
        `detector_client` / `judge_client` — pass distinct clients (e.g. pointed
        at two different llama-server instances/ports) to run detection and
        judging against different models.
        """
        self.detector_client = detector_client or client
        self.judge_client = judge_client or client
        if self.detector_client is None or self.judge_client is None:
            raise ValueError(
                "Provide either `client` (used for both roles) or both "
                "`detector_client` and `judge_client`."
            )
        self.detector_model = detector_model
        self.judge_model = judge_model
        self.max_rounds = max_rounds
        self.score_threshold = score_threshold
        self.detector_template = detector_template
        self.judge_template = judge_template
        self.detector_max_tokens = detector_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.api_retries = api_retries
        self.detector_temperature = detector_temperature
        self.detector_top_p = detector_top_p
        self.judge_temperature = judge_temperature
        self.preprocessing_config = preprocessing_config or {}

    def get_detector_prompt(self, categories, category_definitions, feedback=None, som_proposals=None):
        feedback_block = ""
        if feedback:
            feedback_block = f"""
## Feedback from a previous attempt on this same image
A separate quality-control reviewer inspected your last attempt and found the issues below.
Correct them in this attempt: add any missed objects, fix wrong labels, tighten or loosen
boxes as needed, and remove false positives or duplicates. Keep everything from the previous
attempt that the reviewer did not flag as wrong.

{feedback}
"""
        som_block = ""
        if som_proposals:
            som_block = "\n\n## Candidate Regions (Set-of-Mark)\n"
            som_block += "The image contains numbered candidate regions. If you detect an object that aligns with one of these regions, you should prefer outputting its coordinates. Below is the list of candidates and their approximate coordinates on a 0-1000 scale:\n"
            for prop in som_proposals:
                som_block += f"- Candidate #{prop['id']}: label proposals around bbox_2d: {prop['bbox_2d']}\n"
            som_block += "\nYou can either refer to these candidates or output standard bounding boxes."

        return self.detector_template.format(
            categories_list=", ".join(categories),
            category_definitions=category_definitions + som_block,
            feedback_block=feedback_block,
        )

    def get_judge_prompt(self, category_definitions, detections):
        return self.judge_template.format(
            category_definitions=category_definitions,
            detections_json=json.dumps(detections),
        )

    def run_inference(self, image_uri, categories, category_definitions, feedback=None, som_proposals=None) -> str:
        prompt = self.get_detector_prompt(categories, category_definitions, feedback, som_proposals)

        def _do_call():
            return self.detector_client.chat.completions.create(
                model=self.detector_model,
                temperature=self.detector_temperature,
                top_p=self.detector_top_p,
                max_tokens=self.detector_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    }
                ],
            )

        response = _call_with_retries(_do_call, retries=self.api_retries, what="Detector call")
        return response.choices[0].message.content

    def verify_crop(self, crop_image: Image.Image, label: str) -> bool:
        """
        Verify if the given target class label is present in the cropped image
        by asking the VLM to perform a second-pass confirmation.
        """
        crop_uri = pil_to_data_uri(crop_image)
        prompt = f"Analyze this image crop carefully. Is there a visible '{label}' present inside this crop? You must respond in exactly this format, with nothing else: <present>YES</present> or <present>NO</present>."

        def _do_call():
            return self.detector_client.chat.completions.create(
                model=self.detector_model,
                temperature=0.1,  # low temp for validation
                max_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": crop_uri}},
                        ],
                    }
                ],
            )

        try:
            response = _call_with_retries(_do_call, retries=self.api_retries, what="Crop verification call")
            text = response.choices[0].message.content.strip()
            logger.info("Verification response for label '%s': %s", label, text)
            match = re.search(r"<present>\s*(YES|NO)\s*</present>", text, re.IGNORECASE)
            if match:
                return match.group(1).upper() == "YES"
            return "YES" in text.upper()
        except Exception as e:
            logger.warning("Crop verification failed for label '%s', keeping detection: %s", label, e)
            return True  # Fallback to keeping it if API fails

    def judge_detections(self, original_grid_uri, annotated_grid_uri, detections, category_definitions):
        prompt = self.get_judge_prompt(category_definitions, detections)

        def _do_call():
            return self.judge_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                max_tokens=self.judge_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "Original image (grid, no boxes):"},
                            {"type": "image_url", "image_url": {"url": original_grid_uri}},
                            {"type": "text", "text": "Annotated image (grid + detected boxes):"},
                            {"type": "image_url", "image_url": {"url": annotated_grid_uri}},
                        ],
                    }
                ],
            )

        response = _call_with_retries(_do_call, retries=self.api_retries, what="Judge call")
        text = response.choices[0].message.content

        score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
        feedback_match = re.search(r"<feedback>(.*?)</feedback>", text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(10, score))
        feedback_text = feedback_match.group(1).strip() if feedback_match else text.strip()

        return score, feedback_text

def _filter_and_translate_feedback_for_tile(feedback: str, tile_x: int, tile_y: int, tile_w: int, tile_h: int, orig_w: int, orig_h: int) -> str:
    if not feedback:
        return ""
    lines = feedback.split("\n")
    new_lines = []
    for line in lines:
        matches = re.findall(r"\((\d+)\s*,\s*(\d+)\)", line)
        if not matches:
            new_lines.append(line)
            continue
        
        keep_line = False
        translated_line = line
        for x_str, y_str in matches:
            x_val = int(x_str)
            y_val = int(y_str)
            px = x_val * orig_w / 1000
            py = y_val * orig_h / 1000
            
            if tile_x <= px <= tile_x + tile_w and tile_y <= py <= tile_y + tile_h:
                keep_line = True
                tx = int(round((px - tile_x) * 1000 / tile_w))
                ty = int(round((py - tile_y) * 1000 / tile_h))
                translated_line = translated_line.replace(f"({x_str},{y_str})", f"({tx},{ty})")
                translated_line = translated_line.replace(f"({x_str}, {y_str})", f"({tx},{ty})")
        
        if keep_line:
            new_lines.append(translated_line)
            
    return "\n".join(new_lines)


    def run(
        self,
        image_path: str,
        categories: list[str],
        category_definitions: str,
        show_plot: bool = True,
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        Runs the object detection pipeline with custom preprocessing, tiling, NMS, SoM,
        and Crop & Verify validation.
        """
        if not categories:
            raise ValueError("`categories` must be a non-empty list.")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 1. Load original image and correct color space / exif rotation
        base_image_raw = Image.open(path)
        base_image_raw = preprocess_color_space(base_image_raw, white_balance=self.preprocessing_config.get("white_balance", False))
        orig_w, orig_h = base_image_raw.size

        # 2. Apply resolution scaling and padding
        preprocessed_image, prep_info = preprocess_resolution(
            base_image_raw,
            enabled=self.preprocessing_config.get("resolution_enabled", False),
            target_short_edge=self.preprocessing_config.get("target_short_edge", 1024),
            pad_to_square=self.preprocessing_config.get("pad_to_square", False)
        )
        prep_w, prep_h = preprocessed_image.size

        # 3. Apply contrast enhancement
        preprocessed_image = preprocess_contrast(
            preprocessed_image,
            method=self.preprocessing_config.get("contrast_method", "none"),
            clip_limit=self.preprocessing_config.get("clip_limit", 2.0),
            gamma=self.preprocessing_config.get("gamma", 1.0)
        )

        # 4. Apply noise filtering and sharpening
        preprocessed_image = preprocess_noise_sharpness(
            preprocessed_image,
            method=self.preprocessing_config.get("denoise_method", "none"),
            sharpen=self.preprocessing_config.get("sharpen", False)
        )

        # 5. Determine grid overlay style
        grid_style = self.preprocessing_config.get("grid_style", "standard")

        feedback = None
        history: list[RoundResult] = []
        best = {"score": -1, "annotated": None, "detections": None, "round": 0}

        tiling_enabled = self.preprocessing_config.get("tiling_enabled", False)
        tile_size = self.preprocessing_config.get("tile_size", 512)
        tile_overlap = self.preprocessing_config.get("tile_overlap", 0.2)

        for round_num in range(1, self.max_rounds + 1):
            logger.info("=== Round %d/%d ===", round_num, self.max_rounds)
            detections_prep = []
            parse_error = None
            raw_outputs_collected = []

            if tiling_enabled:
                logger.info("Tiling enabled: dividing image of size %dx%d into tiles of size %d", prep_w, prep_h, tile_size)
                tiles = get_image_tiles(preprocessed_image, tile_size=tile_size, overlap_pct=tile_overlap)
                logger.info("Generated %d tiles", len(tiles))
                
                all_tile_detections = []
                for idx, tile in enumerate(tiles, 1):
                    # Filter feedback coordinates for this tile local system
                    tile_feedback = _filter_and_translate_feedback_for_tile(
                        feedback,
                        tile_x=tile["tile_x"],
                        tile_y=tile["tile_y"],
                        tile_w=tile["tile_w"],
                        tile_h=tile["tile_h"],
                        orig_w=prep_w,
                        orig_h=prep_h
                    ) if feedback else None

                    tile_img_with_grid = draw_premium_grid(tile["tile_image"], style=grid_style)
                    tile_uri = pil_to_data_uri(tile_img_with_grid)

                    logger.info("Running detection on Tile %d/%d (at x=%d, y=%d)...", idx, len(tiles), tile["tile_x"], tile["tile_y"])
                    try:
                        tile_raw_text = self.run_inference(
                            image_uri=tile_uri,
                            categories=categories,
                            category_definitions=category_definitions,
                            feedback=tile_feedback,
                            som_proposals=None
                        )
                        raw_outputs_collected.append(f"Tile {idx} (x={tile['tile_x']}, y={tile['tile_y']}):\n{tile_raw_text}")
                        tile_dets = validate_detections(parse_detections(tile_raw_text), categories)
                        
                        # Map tile local detections back to full preprocessed scale (0-1000)
                        for det in tile_dets:
                            mapped = map_tile_detection_to_original(
                                det["bbox_2d"],
                                tile_x=tile["tile_x"],
                                tile_y=tile["tile_y"],
                                tile_w=tile["tile_w"],
                                tile_h=tile["tile_h"],
                                orig_w=prep_w,
                                orig_h=prep_h
                            )
                            det["bbox_2d"] = mapped
                            all_tile_detections.append(det)
                    except Exception as exc:
                        logger.error("Failed detection on tile %d: %s", idx, exc)
                        parse_error = str(exc) if not parse_error else parse_error + f"; Tile {idx}: {exc}"

                # Merge tile detections using Non-Maximum Suppression
                detections_prep = apply_nms(all_tile_detections, iou_threshold=0.5)
                raw_text = "\n\n".join(raw_outputs_collected)
            else:
                # Full-image processing path
                # Overlay Grid
                grid_img = draw_premium_grid(preprocessed_image, style=grid_style)
                
                # Overlay Set-of-Mark proposals if enabled
                som_proposals = None
                if self.preprocessing_config.get("som_enabled", False):
                    logger.info("Set-of-Mark (SoM) prompting enabled. Generating candidate regions...")
                    grid_img, som_proposals = generate_som_proposals(grid_img)
                    logger.info("Generated %d candidate proposal regions", len(som_proposals))
                
                grid_uri = pil_to_data_uri(grid_img)
                raw_text = self.run_inference(
                    image_uri=grid_uri,
                    categories=categories,
                    category_definitions=category_definitions,
                    feedback=feedback,
                    som_proposals=som_proposals
                )
                
                try:
                    detections_prep = validate_detections(parse_detections(raw_text), categories)
                except ValueError as exc:
                    logger.error("Detector output parsing failed: %s", exc)
                    logger.debug(traceback.format_exc())
                    detections_prep = []
                    parse_error = str(exc)

            # 6. Apply Crop & Verify second-pass validation if enabled
            if detections_prep and self.preprocessing_config.get("crop_verify_enabled", False):
                crop_padding = self.preprocessing_config.get("crop_padding", 0.15)
                logger.info("Crop & Verify validation enabled. Validating %d detections...", len(detections_prep))
                
                verified_detections = []
                
                def verify_single(det):
                    x1, y1, x2, y2 = det["bbox_2d"]
                    px1 = x1 * prep_w / 1000
                    py1 = y1 * prep_h / 1000
                    px2 = x2 * prep_w / 1000
                    py2 = y2 * prep_h / 1000
                    
                    pw = px2 - px1
                    ph = py2 - py1
                    pad_w = pw * crop_padding
                    pad_h = ph * crop_padding
                    
                    cx1 = max(0, int(px1 - pad_w))
                    cy1 = max(0, int(py1 - pad_h))
                    cx2 = min(prep_w, int(px2 + pad_w))
                    cy2 = min(prep_h, int(py2 + pad_h))
                    
                    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                        return det, True
                        
                    crop_img = preprocessed_image.crop((cx1, cy1, cx2, cy2))
                    is_valid = self.verify_crop(crop_img, det["label"])
                    return det, is_valid

                with ThreadPoolExecutor(max_workers=4) as executor:
                    verification_results = list(executor.map(verify_single, detections_prep))
                    
                for det, is_valid in verification_results:
                    if is_valid:
                        verified_detections.append(det)
                    else:
                        logger.info("Crop & Verify: discarded detection box %s for label '%s'", det["bbox_2d"], det["label"])
                        
                detections_prep = verified_detections

            # 7. Map coordinates from preprocessed scale back to the original image scale
            detections_orig = []
            for det in detections_prep:
                mapped_box = map_bbox_to_original(det["bbox_2d"], prep_info)
                detections_orig.append({"label": det["label"], "bbox_2d": mapped_box})

            # Render detections on the original scale base image for visualization
            annotated_orig = render_detections(base_image_raw, detections_orig)
            
            # Draw preprocessed annotated view with grid for the judge
            annotated_prep = render_detections(preprocessed_image, detections_prep)
            annotated_prep_with_grid = draw_premium_grid(annotated_prep, style=grid_style)
            annotated_prep_uri = pil_to_data_uri(annotated_prep_with_grid)
            
            # Setup original scale background with grid for the judge
            grid_original_prep = draw_premium_grid(preprocessed_image, style=grid_style)
            grid_original_prep_uri = pil_to_data_uri(grid_original_prep)

            score, judge_feedback = self.judge_detections(
                original_grid_uri=grid_original_prep_uri,
                annotated_grid_uri=annotated_prep_uri,
                detections=detections_prep,
                category_definitions=category_definitions,
            )

            logger.info("Judge score: %d/10", score)
            logger.info("Judge feedback:\n%s", judge_feedback)

            round_result = RoundResult(
                round=round_num,
                detections=detections_orig,
                score=score,
                feedback=judge_feedback,
                raw_detector_output=raw_text,
                parse_error=parse_error,
            )
            history.append(round_result)

            if progress_callback:
                try:
                    progress_callback(round_result, annotated_orig)
                except Exception:
                    logger.warning("progress_callback raised an exception", exc_info=True)

            if score > best["score"]:
                best = {"score": score, "annotated": annotated_orig, "detections": detections_orig, "round": round_num}

            if score >= self.score_threshold:
                logger.info("Score threshold (%d) reached at round %d, stopping.", self.score_threshold, round_num)
                break

            feedback = judge_feedback

        logger.info("Best result: round %d with score %d/10", best["round"], best["score"])

        if output_dir:
            self._persist(output_dir, base_image_raw, best, history)

        if show_plot and best["annotated"] is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(best["annotated"])
            plt.axis("off")
            plt.title(f"Best detections (round {best['round']}, score {best['score']}/10)")
            plt.show()

        return best, history

    @staticmethod
    def _persist(output_dir: str, base_image: Image.Image, best: dict, history: list[RoundResult]):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if best["annotated"] is not None:
            best["annotated"].save(out / "best_annotated.jpg")

        (out / "best_detections.json").write_text(json.dumps(best["detections"], indent=2))

        history_payload = [
            {
                "round": r.round,
                "score": r.score,
                "detections": r.detections,
                "feedback": r.feedback,
                "parse_error": r.parse_error,
            }
            for r in history
        ]
        (out / "history.json").write_text(json.dumps(history_payload, indent=2))
        logger.info("Persisted results to %s", out.resolve())


# ---------------------------------------------------------------------------
# Backward-compat alias (keeps old import names working)
# ---------------------------------------------------------------------------

FabricDefectPipeline = ObjectDetectionPipeline


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_client = OpenAI(
        api_key="not-needed",
        base_url="http://localhost:8080/v1",
    )

    # Example: general object detection
    categories = ["person", "car", "bicycle", "dog", "cat"]
    definitions = """
- person: a human being
- car: a 4-wheeled motor vehicle
- bicycle: a 2-wheeled human-powered vehicle
- dog: a domestic canine
- cat: a domestic feline
"""
    image_path = "/path/to/your/image.jpg"

    pipeline = ObjectDetectionPipeline(
        client=api_client,
        detector_model="local-model",
        judge_model="local-model",
        max_rounds=2,
        score_threshold=8,
    )

    best_res, run_hist = pipeline.run(
        image_path=image_path,
        categories=categories,
        category_definitions=definitions,
        output_dir="./detection_output",
    )
