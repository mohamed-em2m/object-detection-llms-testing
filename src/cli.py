"""
CLI entry point for the LLM Object Detection tester.

Run a detection pipeline against one or more images from the command line.

Usage:
    uv run detection-cli --help
    uv run detection-cli --image path/to/img.jpg --categories "person, car, dog"
    uv run detection-cli --image img1.jpg --image img2.jpg \\
        --categories "cat, dog" \\
        --definitions "cat: a feline; dog: a canine" \\
        --base-url http://localhost:8080/v1 \\
        --detector-model local-model \\
        --judge-model local-model \\
        --max-rounds 3 \\
        --score-threshold 8 \\
        --output-dir ./results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from detection_pipeline import ObjectDetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detection-cli",
        description="Run the LLM object-detection pipeline on one or more images from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Images ---
    p.add_argument(
        "--image", "-i",
        metavar="PATH",
        action="append",
        required=True,
        dest="images",
        help="Path to an input image. Can be specified multiple times.",
    )

    # --- Categories ---
    p.add_argument(
        "--categories", "-c",
        metavar="LIST",
        default="person, car, bicycle, dog, cat",
        help="Comma-separated list of object categories to detect.",
    )
    p.add_argument(
        "--definitions", "-d",
        metavar="TEXT",
        default="",
        help="Optional category definitions (plain text, one per line).",
    )

    # --- Server / model ---
    p.add_argument(
        "--base-url",
        metavar="URL",
        default="http://localhost:8080/v1",
        help="OpenAI-compatible base URL of the inference server.",
    )
    p.add_argument(
        "--api-key",
        metavar="KEY",
        default="not-needed",
        help="API key (use 'not-needed' for local servers).",
    )
    p.add_argument(
        "--detector-model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the detector API request.",
    )
    p.add_argument(
        "--judge-model",
        metavar="NAME",
        default="local-model",
        help="Model name sent in the judge API request.",
    )
    p.add_argument(
        "--judge-url",
        metavar="URL",
        default=None,
        help="Separate base URL for the judge model (defaults to --base-url).",
    )

    # --- Pipeline params ---
    p.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum number of detector/judge rounds per image.",
    )
    p.add_argument(
        "--score-threshold",
        type=int,
        default=8,
        help="Stop early when judge score reaches this value (0–10).",
    )
    p.add_argument(
        "--detector-temperature",
        type=float,
        default=0.9,
    )
    p.add_argument(
        "--detector-top-p",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--judge-temperature",
        type=float,
        default=0.2,
    )
    p.add_argument(
        "--detector-max-tokens",
        type=int,
        default=4096,
    )
    p.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--api-retries",
        type=int,
        default=3,
    )

    # --- Preprocessing ---
    p.add_argument("--prep-enabled", action="store_true", help="Enable image preprocessing.")
    p.add_argument("--prep-short-edge", type=int, default=1024, help="Target size for short edge of the image.")
    p.add_argument("--prep-pad-square", action="store_true", help="Pad preprocessed image to square with neutral gray.")
    p.add_argument("--prep-contrast-method", choices=["none", "clahe", "autocontrast"], default="none", help="Contrast enhancement method.")
    p.add_argument("--prep-gamma", type=float, default=1.0, help="Gamma correction factor.")
    p.add_argument("--prep-denoise-method", choices=["none", "bilateral", "nlm"], default="none", help="Denoising method.")
    p.add_argument("--prep-sharpen", action="store_true", help="Apply unsharp mask sharpening.")
    p.add_argument("--prep-white-balance", action="store_true", help="Apply white balance correction.")
    p.add_argument("--prep-grid-style", choices=["standard", "transparent", "fine", "none"], default="standard", help="Visual grid overlay style.")
    p.add_argument("--prep-som-enabled", action="store_true", help="Enable Set-of-Mark visual prompting overlay.")
    p.add_argument("--prep-tiling-enabled", action="store_true", help="Enable image tiling for small object detection.")
    p.add_argument("--prep-tile-size", type=int, default=512, help="Tile size in pixels.")
    p.add_argument("--prep-tile-overlap", type=float, default=0.2, help="Overlap ratio between tiles (0.0 to 0.5).")
    p.add_argument("--prep-crop-verify-enabled", action="store_true", help="Enable multi-pass Crop & Verify validation pipeline.")
    p.add_argument("--prep-crop-padding", type=float, default=0.15, help="Context padding ratio for cropped patches.")

    # --- Output ---
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default="./detection_results",
        help="Base output directory. Each image gets its own sub-folder.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib preview window.",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build clients
    detector_client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    judge_url = args.judge_url or args.base_url
    judge_client = (
        OpenAI(api_key=args.api_key, base_url=judge_url)
        if judge_url != args.base_url
        else detector_client
    )

    # Parse categories
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        print("ERROR: --categories must contain at least one entry.", file=sys.stderr)
        sys.exit(1)

    # Build preprocessing config
    if not args.prep_enabled:
        prep_config = {
            "resolution_enabled": False,
            "contrast_method": "none",
            "denoise_method": "none",
            "som_enabled": False,
            "tiling_enabled": False,
            "crop_verify_enabled": False,
            "grid_style": args.prep_grid_style,
        }
    else:
        prep_config = {
            "resolution_enabled": True,
            "target_short_edge": args.prep_short_edge,
            "pad_to_square": args.prep_pad_square,
            "contrast_method": args.prep_contrast_method,
            "clip_limit": 2.0,
            "gamma": args.prep_gamma,
            "denoise_method": args.prep_denoise_method,
            "sharpen": args.prep_sharpen,
            "white_balance": args.prep_white_balance,
            "grid_style": args.prep_grid_style,
            "som_enabled": args.prep_som_enabled,
            "tiling_enabled": args.prep_tiling_enabled,
            "tile_size": args.prep_tile_size,
            "tile_overlap": args.prep_tile_overlap,
            "crop_verify_enabled": args.prep_crop_verify_enabled,
            "crop_padding": args.prep_crop_padding,
        }

    # Build pipeline
    pipeline = ObjectDetectionPipeline(
        detector_client=detector_client,
        judge_client=judge_client,
        detector_model=args.detector_model,
        judge_model=args.judge_model,
        max_rounds=args.max_rounds,
        score_threshold=args.score_threshold,
        detector_temperature=args.detector_temperature,
        detector_top_p=args.detector_top_p,
        judge_temperature=args.judge_temperature,
        detector_max_tokens=args.detector_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        api_retries=args.api_retries,
        preprocessing_config=prep_config,
    )

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    all_results = []

    for image_path in args.images:
        p = Path(image_path)
        if not p.is_file():
            print(f"WARNING: image not found, skipping: {image_path}", file=sys.stderr)
            continue

        image_out_dir = out_base / p.stem
        print(f"\n{'='*60}")
        print(f"Processing: {p.name}  →  {image_out_dir}")
        print(f"{'='*60}")

        def on_round(round_result, _annotated):
            print(
                f"  Round {round_result.round}: "
                f"score {round_result.score}/10, "
                f"{len(round_result.detections)} detection(s)"
                + (f"  [parse error]" if round_result.parse_error else "")
            )

        try:
            best, history = pipeline.run(
                image_path=str(p),
                categories=categories,
                category_definitions=args.definitions,
                show_plot=not args.no_plot,
                output_dir=str(image_out_dir),
                progress_callback=on_round,
            )
            print(f"  ✅ Best: round {best['round']}, score {best['score']}/10, "
                  f"{len(best['detections'] or [])} detection(s)")
            all_results.append({
                "image": str(p),
                "status": "ok",
                "best_round": best["round"],
                "best_score": best["score"],
                "n_detections": len(best["detections"] or []),
                "output_dir": str(image_out_dir),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ ERROR: {exc}", file=sys.stderr)
            all_results.append({"image": str(p), "status": f"error: {exc}"})

    # Write summary
    summary_path = out_base / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
