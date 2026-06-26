# 🔍 LLM Object Detection Testing Console

An interactive test console and command-line tool for assessing and refining Vision-Language Models (VLMs) on general Object Detection tasks. 

This project implements an iterative **Detector-Judge pipeline** that prompts a VLM "detector" agent to locate objects, lets a VLM "judge" agent critique the proposed boxes against the original image, and repeats the loop with structured feedback until the annotations meet a quality threshold or the round limit is reached.

---

## 🌟 Key Features

* **Iterative Refinement Loop**: Enhances detection accuracy by feeding visual and text-based critiques back to the detector across multiple rounds.
* **Advanced Image Preprocessing & Augmentation Pipeline**:
  * **Dynamic Resolution Tuning**: Custom upscaling target for short edge and optional letterbox padding to maintain aspect ratio on square inputs.
  * **Contrast & Color Enhancements**: LAB color-space CLAHE, global autocontrast, gamma correction (0.5–2.0), and Gray World white balance correction.
  * **Noise Filtering & Sharpening**: Bilateral or Non-Local Means (NLM) denoising with edge preservation, and unsharp mask sharpening.
  * **Tiling Engine for Small Objects**: Slices high-resolution inputs into overlapping sub-patches, runs sequential detection on each tile, and merges overlapping detections using Non-Maximum Suppression (NMS).
  * **Crop & Verify Validation**: Runs a second-pass confirmation on cropped candidate coordinates with context padding to reduce hallucinated detections.
  * **Set-of-Mark (SoM) Prompting**: Classic CV contour detection overlays numbered candidate regions onto the image to convert the spatial regression task into a simpler classification/selection task.
  * **Fully Customizable Grid Overlays**: Adjustable style (standard, transparent, fine, none), step size (divisions), line thickness, font size, and custom grid color palettes (line, text, and text-backing box colors with CSS name or hex inputs).
  * **VLM Processor Pixel Bounds**: Manually configure `min_pixels` and `max_pixels` request parameters (passed in the API request's `extra_body` payload) to tune vision encoder resolution and prevent OOMs on backends like vLLM / Qwen-VL.
* **Gradio Web Interface**:
  * **🦙 Llama Server**: Start, stop, and configure multiple local `llama-server` instances on different ports, track logs, and select the active server.
  * **⚙️ Detection Settings**: Customize target categories, category definitions, prompt templates, per-role model/server routing, and sampling parameters.
  * **🧪 Batch Test**: Upload multiple images, run them sequentially, view live per-round annotated results, and download all outputs as a `.zip` archive.
  * **🖼️ Interactive Preprocessing & Custom Grids**: Tweak all resolution scaling, filters, tiling options, custom grid colors, text sizes, and pixel bounds directly through visual control elements.
* **Command-Line Interface (CLI)**: Run the object detection pipeline on single or multiple images directly from your terminal.
* **Visual Annotations & Grids**: Automatically overlays a customizable 0-1000 coordinate grid on images to aid the VLM's spatial reasoning, and draws lime-green bounding boxes with legible text labels for final detections.
* **Robust File Persistence**: Saves the best annotated image, raw JSON detections, and complete round-by-round history for every processed image.

---

## 🚀 Setup & Installation

This project is managed with [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamed-em2m/llm-object-grounding.git
   cd llm-object-grounding
   ```

2. **Install dependencies**:
   ```bash
   ./scripts/install_llama_cpp.sh
   uv sync
   ```

---

## 🖥️ Usage

### 1. Launching the Web GUI
To launch the interactive Gradio interface:
```bash
uv run detection-gui
```
Options:
* `--host`: Host to bind the Gradio server to (default: `0.0.0.0`).
* `--port`: Port to run the server on (default: `7860`).
* `--share`: Create a public Gradio share link.
* `--no-queue`: Disable Gradio's request queue.

Example:
```bash
uv run detection-gui --port 7861 --share
```

### 2. Running the Command-Line Interface (CLI)
You can run the detection pipeline directly from the command line:
```bash
uv run detection-cli --image path/to/image.jpg --categories "person, car, dog"
```

#### CLI Options:
* `-i`, `--image`: Path to the input image (can be specified multiple times for batch processing).
* `-c`, `--categories`: Comma-separated list of object categories to detect (default: `person, car, bicycle, dog, cat`).
* `-d`, `--definitions`: Optional category definitions to help the VLM distinguish similar categories.
* `--base-url`: OpenAI-compatible API base URL (default: `http://localhost:8080/v1`).
* `--detector-model` / `--judge-model`: Models to use for detection and judging.
* `--max-rounds`: Max detector-judge iterations per image (default: `2`).
* `--score-threshold`: Quality score (0-10) to stop the loop early (default: `8`).
* `--output-dir`: Output directory to save the results (default: `./detection_results`).
* `--no-plot`: Skip displaying the matplotlib preview window after completion.

##### Preprocessing Pipeline options:
* `--prep-enabled`: Enable image preprocessing.
* `--prep-short-edge`: Target size for short edge of the image (default: `1024`).
* `--prep-pad-square`: Pad preprocessed image to square with neutral gray.
* `--prep-contrast-method`: Contrast enhancement method (`none`, `clahe`, `autocontrast`).
* `--prep-gamma`: Gamma correction factor (default: `1.0`).
* `--prep-denoise-method`: Denoising method (`none`, `bilateral`, `nlm`).
* `--prep-sharpen`: Apply unsharp mask sharpening.
* `--prep-white-balance`: Apply white balance correction.
* `--prep-som-enabled`: Enable Set-of-Mark visual prompting overlay.
* `--prep-tiling-enabled`: Enable image tiling for small object detection.
* `--prep-tile-size`: Tile size in pixels (default: `512`).
* `--prep-tile-overlap`: Overlap ratio between tiles (default: `0.2`).
* `--prep-crop-verify-enabled`: Enable multi-pass Crop & Verify validation pipeline.
* `--prep-crop-padding`: Context padding ratio for cropped patches (default: `0.15`).

##### Custom Grid & VLM Pixel Bounds options:
* `--prep-grid-style`: Visual grid overlay style (`standard`, `transparent`, `fine`, `none` - default: `standard`).
* `--prep-grid-step`: Grid line separation on a 0-1000 scale (default: `100`).
* `--prep-grid-line-width`: Grid line thickness in pixels (default: `1`).
* `--prep-grid-font-size`: Grid text label font size (default: `0` / auto).
* `--prep-grid-line-color`: Grid line color - CSS color name or hex string (default: `red`).
* `--prep-grid-text-color`: Grid text label color - CSS color name or hex string (default: `white`).
* `--prep-grid-backing-color`: Grid text label backing box color - CSS color name, hex, or `'none'` (default: `black`).
* `--prep-send-pixel-bounds`: Send min_pixels and max_pixels in OpenAI-compatible API request payload.
* `--prep-min-pixels`: VLM min_pixels parameter passed to model processor (default: `200704`).
* `--prep-max-pixels`: VLM max_pixels parameter passed to model processor (default: `4194304`).

Example running CLAHE, NMS tiling, custom blue coordinates, and custom Qwen-VL model bounds:
```bash
uv run detection-cli \
  -i industrial_part.jpg \
  -c "crack, scratch, dent" \
  --prep-enabled \
  --prep-contrast-method clahe \
  --prep-tiling-enabled \
  --prep-tile-size 512 \
  --prep-tile-overlap 0.25 \
  --prep-grid-line-color "blue" \
  --prep-grid-step 50 \
  --prep-send-pixel-bounds \
  --prep-min-pixels 200704 \
  --prep-max-pixels 2097152 \
  --output-dir ./inspection_results
```

---

## 📁 Output Structure

For every image processed, a dedicated subdirectory is created under the output folder:
```
detection_results/
└── [image_name]/
    ├── best_annotated.jpg    # Best annotated image with lime-green bounding boxes
    ├── best_detections.json  # Final parsed JSON detections
    └── history.json          # Complete round-by-round scores, detections, and feedback
```

---

## 🛠️ Technology Stack

* **Core Logic**: Python 3.12+, Pillow (PIL) for image manipulations, OpenCV (`opencv-python`) for CLAHE/Contour/Bilateral processing, Matplotlib for rendering.
* **VLM Integrations**: OpenAI Python SDK (compatible with any OpenAI-style endpoint such as `llama-server`, vLLM, Ollama, etc. Supports custom payloads via `extra_body`).
* **Web UI**: Gradio.
* **Environment & Package Management**: `uv`, Setuptools.
