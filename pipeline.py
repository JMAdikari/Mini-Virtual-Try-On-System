"""
Phase 3 — Inpainting Pipeline (Replicate API)
==============================================
Takes the image and mask from Phase 2 and generates a new image
with the clothing replaced using Stable Diffusion inpainting.

Usage:
    python pipeline.py samples/input.jpg "black oversized hoodie"
"""

import os
import sys
import io
import time
import base64
import requests
from PIL import Image
from dotenv import load_dotenv

from segmentation import get_clothing_mask, visualize_mask_overlay

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL     = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
OUTPUT_DIR          = "samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64 data URI string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _download_image(url: str) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _save_comparison(original: Image.Image, result: Image.Image, prompt: str) -> str:
    """Save a side-by-side before/after comparison image."""
    w, h = original.size
    gap = 20
    label_height = 40

    comparison = Image.new("RGB", (w * 2 + gap, h + label_height), (30, 30, 30))
    comparison.paste(original, (0, 0))
    comparison.paste(result, (w + gap, 0))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_prompt = prompt[:40].replace(" ", "_").replace("/", "-")
    out_path = os.path.join(OUTPUT_DIR, f"comparison_{safe_prompt}.png")
    comparison.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(user_input: str) -> tuple[str, str]:
    """
    Wrap the user's clothing description in a photorealism template.

    Returns:
        (prompt, negative_prompt)
    """
    prompt = (
        f"a person wearing {user_input}, "
        "photorealistic, studio lighting, high quality, "
        "sharp focus, natural fabric texture, 8k"
    )
    negative_prompt = (
        "cartoon, anime, illustration, blurry, low quality, "
        "deformed, extra limbs, bad anatomy, watermark, text, "
        "duplicate, missing limbs, distorted face"
    )
    return prompt, negative_prompt


# ---------------------------------------------------------------------------
# Replicate API call
# ---------------------------------------------------------------------------

def run_inpainting(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """
    Call the Replicate inpainting API and return the generated image.

    Args:
        image:               Original person photo (512x512 PIL)
        mask:                Binary clothing mask (512x512 PIL, white = replace)
        prompt:              What to generate in the masked area
        negative_prompt:     What to avoid generating
        num_inference_steps: More steps = better quality, slower (default 50)
        guidance_scale:      How strongly to follow the prompt (default 7.5)

    Returns:
        Generated PIL image
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError(
            "REPLICATE_API_TOKEN not found.\n"
            "Add it to your .env file:\n"
            "  REPLICATE_API_TOKEN=r8_xxxxxxxxxxxx"
        )

    print("[pipeline] Converting images to base64...")
    image_b64 = _pil_to_base64(image)
    mask_b64  = _pil_to_base64(mask)

    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "version": REPLICATE_MODEL.split(":")[1],
        "input": {
            "prompt":              prompt,
            "negative_prompt":     negative_prompt,
            "image":               image_b64,
            "mask":                mask_b64,
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
            "width":               512,
            "height":              512,
        }
    }

    print("[pipeline] Sending request to Replicate API...")
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 201:
        raise RuntimeError(
            f"Replicate API error {response.status_code}: {response.text}"
        )

    prediction = response.json()
    prediction_id = prediction["id"]
    print(f"[pipeline] Prediction ID: {prediction_id}")
    print("[pipeline] Waiting for result (this takes 30-60 seconds)...")

    # Poll until complete
    poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    while True:
        poll = requests.get(poll_url, headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data["status"]

        if status == "succeeded":
            output_url = data["output"][0]
            print(f"[pipeline] Generation complete!")
            break
        elif status == "failed":
            raise RuntimeError(
                f"Replicate prediction failed: {data.get('error', 'Unknown error')}"
            )
        else:
            print(f"[pipeline] Status: {status}... waiting")
            time.sleep(3)

    print("[pipeline] Downloading result...")
    result = _download_image(output_url)
    return result


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, clothing_prompt: str) -> tuple[Image.Image, Image.Image]:
    """
    Full pipeline: segment -> mask -> inpaint -> save outputs.

    Args:
        image_path:      Path to person photo
        clothing_prompt: e.g. "black oversized hoodie"

    Returns:
        (original_image, result_image) — both PIL images
    """
    # Step 1 — Phase 2: get image + mask
    print(f"\n[pipeline] Step 1: Generating clothing mask...")
    image, mask = get_clothing_mask(image_path, save_debug=False)

    # Step 2 — Build prompt
    prompt, negative_prompt = build_prompt(clothing_prompt)
    print(f"\n[pipeline] Step 2: Prompt ready")
    print(f"  Prompt  : {prompt}")
    print(f"  Negative: {negative_prompt}")

    # Step 3 — Run inpainting
    print(f"\n[pipeline] Step 3: Running inpainting via Replicate...")
    result = run_inpainting(image, mask, prompt, negative_prompt)

    # Step 4 — Save all outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_path = os.path.join(OUTPUT_DIR, "output.png")
    result.save(result_path)
    print(f"\n[pipeline] Saved: {result_path}")

    comparison_path = _save_comparison(image, result, clothing_prompt)
    print(f"[pipeline] Saved: {comparison_path}")

    overlay = visualize_mask_overlay(image, mask)
    overlay_path = os.path.join(OUTPUT_DIR, "mask_overlay.png")
    overlay.save(overlay_path)
    print(f"[pipeline] Saved: {overlay_path}")

    return image, result


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:   python pipeline.py <image_path> \"<clothing prompt>\"")
        print("Example: python pipeline.py samples/input.jpg \"black oversized hoodie\"")
        print("Example: python pipeline.py samples/input.jpg \"red floral summer dress\"")
        sys.exit(1)

    image_path      = sys.argv[1]
    clothing_prompt = sys.argv[2]

    print("\n" + "=" * 55)
    print("  Mini Virtual Try-On — Phase 3: Inpainting")
    print("=" * 55)
    print(f"  Image  : {image_path}")
    print(f"  Prompt : {clothing_prompt}")
    print("=" * 55)

    original, result = run_pipeline(image_path, clothing_prompt)

    print("\n" + "=" * 55)
    print("  Done! Check your samples/ folder:")
    print("  output.png          -> generated result")
    print("  mask_overlay.png    -> clothing mask preview")
    print("  comparison_*.png    -> before vs after")
    print("=" * 55)