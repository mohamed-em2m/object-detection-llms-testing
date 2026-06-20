# 🔍 LLM Object Detection Testing Console

An interactive test console and command-line tool for assessing and refining Vision-Language Models (VLMs) on general Object Detection tasks. 

This project implements an iterative **Detector-Judge pipeline** that prompts a VLM "detector" agent to locate objects, lets a VLM "judge" agent critique the proposed boxes against the original image, and repeats the loop with structured feedback until the annotations meet a quality threshold or the round limit is reached.

---

## 🌟 Key Features

* **Iterative Refinement Loop**: Enhances detection accuracy by feeding visual and text-based critiques back to the detector across multiple rounds.
* **Gradio Web Interface**:
  * **🦙 Llama Server**: Start, stop, and configure multiple local `llama-server` instances on different ports, track logs, and select the active server.
  * **⚙️ Detection Settings**: Customize target categories, category definitions, prompt templates, per-role model/server routing, and sampling parameters.
  * **🧪 Batch Test**: Upload multiple images, run them sequentially, view live per-round annotated results, and download all outputs as a `.zip` archive.
* **Command-Line Interface (CLI)**: Run the object detection pipeline on single or multiple images directly from your terminal.
* **Visual Annotations & Grids**: Automatically overlays a 0-1000 red coordinate grid on images to aid the VLM's spatial reasoning, and draws lime-green bounding boxes with legible text labels for final detections.
* **Robust File Persistence**: Saves the best annotated image, raw JSON detections, and complete round-by-round history for every processed image.

---

## 🚀 Setup & Installation

This project is managed with [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamed-em2m/object-detection-llms-testing.git
   cd object-detection-llms-testing
   ```

2. **Install dependencies**:
   ```bash
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

Example with multiple images and definitions:
```bash
uv run detection-cli \
  -i img1.jpg -i img2.jpg \
  -c "car, truck, traffic light" \
  -d "car: 4-wheeled passenger vehicle; truck: large cargo vehicle" \
  --max-rounds 3 \
  --output-dir ./my_results
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

* **Core Logic**: Python 3.12+, Pillow (PIL) for image manipulations, Matplotlib for rendering.
* **VLM Integrations**: OpenAI Python SDK (compatible with any OpenAI-style endpoint such as `llama-server`, vLLM, Ollama, etc.).
* **Web UI**: Gradio.
* **Environment & Package Management**: `uv`, Setuptools.
