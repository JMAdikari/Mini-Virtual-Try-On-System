"""
Phase 3 — Inpainting Pipeline (Replicate API)
==============================================
Auto-detects clothing category from the user's prompt:
  - "jeans", "pants", "skirt" etc. → lower body mask
  - "dress", "jumpsuit" etc.       → full body mask
  - everything else                → upper body mask (default)

Strength is tuned per category:
  - upper: 0.55 (preserve pose tightly)
  - lower: 0.75 (need more freedom for color changes)
  - full:  0.75 (full body replacement needs more freedom)

Usage:
    python pipeline.py samples/input.jpg "blue oversized hoodie"
    python pipeline.py samples/input.jpg "red jeans"
    python pipeline.py samples/input.jpg "floral summer dress"
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

# Strength per category
# Lower values = model stays closer to original (good for pose preservation)
# Higher values = model has more freedom (needed for dramatic color/style changes)
STRENGTH_MAP = {
    "upper": 0.55,
    "lower": 0.75,
    "full":  0.75,
}


# ---------------------------------------------------------------------------
# Category auto-detection
# ---------------------------------------------------------------------------

LOWER_BODY_KEYWORDS = [
    "pants", "jeans", "trousers", "chinos", "shorts",
    "skirt", "leggings", "joggers", "sweatpants",
    "cargo pants", "flare pants", "wide leg pants",
    "culottes", "slacks", "khakis",
]

FULL_BODY_KEYWORDS = [
    "dress", "gown", "jumpsuit", "romper", "overalls",
    "bodysuit", "maxi", "mini dress", "sundress",
    "outfit", "full outfit", "suit",
]

def detect_category(prompt: str) -> str:
    """
    Auto-detect clothing category from the prompt.

    Returns:
        "upper"  — shirts, hoodies, jackets, sweaters (default)
        "lower"  — pants, jeans, skirts, shorts
        "full"   — dresses, jumpsuits, full outfits
    """
    p = prompt.lower()

    # Full body takes priority over lower body
    if any(k in p for k in FULL_BODY_KEYWORDS):
        print(f"[pipeline] Detected category: FULL BODY")
        return "full"

    if any(k in p for k in LOWER_BODY_KEYWORDS):
        print(f"[pipeline] Detected category: LOWER BODY")
        return "lower"

    print(f"[pipeline] Detected category: UPPER BODY (default)")
    return "upper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _save_comparison(original: Image.Image, result: Image.Image, prompt: str) -> str:
    w, h = original.size
    comparison = Image.new("RGB", (w * 2 + 20, h), (15, 15, 15))
    comparison.paste(original, (0, 0))
    comparison.paste(result, (w + 20, 0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = prompt[:40].replace(" ", "_").replace("/", "-")
    path = os.path.join(OUTPUT_DIR, f"comparison_{safe}.png")
    comparison.save(path)
    return path


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(clothing_prompt: str, category: str) -> tuple:
    """
    Build prompt and negative prompt tuned per category.
    Lower body prompts repeat the clothing description and
    explicitly block original colors for better color fidelity.
    """

    if category == "lower":
        # Extract color from prompt if present for stronger negative
        common_original_colors = "dark jeans, navy jeans, blue jeans, black jeans, dark pants, navy pants"

        prompt = (
            f"person wearing {clothing_prompt}, "
            f"{clothing_prompt} on legs and lower body, "
            "same upper body, same shirt, same top, same jacket, "
            "same face, same background, same pose, "
            "photorealistic, high quality, sharp focus, natural fabric texture"
        )
        negative_prompt = (
            f"{common_original_colors}, "
            "different upper body, changed shirt, changed top, "
            "different person, different face, different pose, "
            "cartoon, anime, blurry, low quality, deformed, "
            "extra limbs, bad anatomy, watermark, text, changed background"
        )

    elif category == "full":
        prompt = (
            f"person wearing {clothing_prompt}, "
            f"{clothing_prompt} as full outfit, "
            "same face, same hairstyle, same background, same pose, "
            "photorealistic, high quality, sharp focus, natural fabric texture"
        )
        negative_prompt = (
            "different person, different face, changed hairstyle, "
            "different pose, cartoon, anime, blurry, low quality, "
            "deformed, extra limbs, bad anatomy, watermark, text, "
            "changed background, distorted face"
        )

    else:  # upper
        prompt = (
            f"person wearing {clothing_prompt}, "
            f"{clothing_prompt} on upper body and torso, "
            "same pants, same lower body, same shoes, "
            "same face, same background, same pose, same body position, "
            "photorealistic, high quality, sharp focus, natural fabric texture"
        )
        negative_prompt = (
            "different pants, changed lower body, different shoes, "
            "different person, different face, different pose, "
            "crossed arms, changed body position, changed hands, "
            "cartoon, anime, blurry, low quality, deformed, "
            "extra limbs, bad anatomy, watermark, text, "
            "distorted face, changed background"
        )

    return prompt, negative_prompt


# ---------------------------------------------------------------------------
# Replicate API
# ---------------------------------------------------------------------------

def run_inpainting(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    original_size: tuple,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    strength: float = 0.55,
) -> Image.Image:
    """
    Call Replicate inpainting API.
    Returns result resized back to original_size.
    """
    if not REPLICATE_API_TOKEN:
        raise ValueError(
            "REPLICATE_API_TOKEN not found.\n"
            "Add to .env: REPLICATE_API_TOKEN=r8_xxxxxxxxxxxx"
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
            "strength":            strength,
            "width":               512,
            "height":              512,
            "num_outputs":         1,
        }
    }

    print(f"[pipeline] Strength: {strength} | Guidance: {guidance_scale}")
    print("[pipeline] Sending to Replicate API...")
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 201:
        raise RuntimeError(f"Replicate API error {response.status_code}: {response.text}")

    prediction_id = response.json()["id"]
    print(f"[pipeline] Prediction ID: {prediction_id}")
    print("[pipeline] Waiting for result (30-60 seconds)...")

    poll_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    while True:
        poll   = requests.get(poll_url, headers=headers, timeout=30)
        poll.raise_for_status()
        data   = poll.json()
        status = data["status"]

        if status == "succeeded":
            print("[pipeline] Generation complete!")
            output_url = data["output"][0]
            break
        elif status == "failed":
            raise RuntimeError(f"Prediction failed: {data.get('error', 'Unknown')}")
        else:
            print(f"[pipeline] Status: {status}... waiting")
            time.sleep(3)

    print("[pipeline] Downloading result...")
    result = _download_image(output_url)

    # Restore original image dimensions
    if result.size != original_size:
        print(f"[pipeline] Resizing {result.size} -> {original_size}")
        result = result.resize(original_size, Image.LANCZOS)

    return result


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, clothing_prompt: str) -> tuple:
    """
    Full pipeline:
      detect category -> segment -> mask -> inpaint -> resize -> save

    Args:
        image_path:      Path to person photo
        clothing_prompt: e.g. "blue oversized hoodie" / "red jeans"

    Returns:
        (original_image, result_image) at original input dimensions
    """
    # Record original size
    original_size = Image.open(image_path).convert("RGB").size
    print(f"[pipeline] Original size: {original_size[0]}x{original_size[1]}px")

    # Step 1 — Detect category
    print(f"\n[pipeline] Step 1: Detecting clothing category...")
    category = detect_category(clothing_prompt)

    # Step 2 — Pick strength based on category
    strength = STRENGTH_MAP[category]
    print(f"[pipeline] Using strength: {strength} for [{category}] category")

    # Step 3 — Generate mask
    print(f"\n[pipeline] Step 2: Generating [{category}] clothing mask...")
    image, mask = get_clothing_mask(image_path, category=category, save_debug=False)

    # Step 4 — Build prompt
    prompt, negative_prompt = build_prompt(clothing_prompt, category)
    print(f"\n[pipeline] Step 3: Prompt ready")
    print(f"  Prompt  : {prompt}")
    print(f"  Negative: {negative_prompt}")

    # Step 5 — Inpaint
    print(f"\n[pipeline] Step 4: Running inpainting...")
    result = run_inpainting(
        image, mask, prompt, negative_prompt,
        original_size,
        strength=strength,
    )

    # Step 6 — Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_original = image.resize(original_size, Image.LANCZOS)

    result_path = os.path.join(OUTPUT_DIR, "output.png")
    result.save(result_path)
    print(f"\n[pipeline] Saved: {result_path}  ({result.size[0]}x{result.size[1]}px)")

    comparison_path = _save_comparison(image_original, result, clothing_prompt)
    print(f"[pipeline] Saved: {comparison_path}")

    overlay = visualize_mask_overlay(image, mask)
    overlay.resize(original_size, Image.LANCZOS).save(
        os.path.join(OUTPUT_DIR, "mask_overlay.png")
    )
    print(f"[pipeline] Saved: {OUTPUT_DIR}/mask_overlay.png")

    return image_original, result


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:   python pipeline.py <image> \"<prompt>\"")
        print("Examples:")
        print('  python pipeline.py samples/input.jpg "blue oversized hoodie"')
        print('  python pipeline.py samples/input.jpg "red jeans"')
        print('  python pipeline.py samples/input.jpg "floral summer dress"')
        sys.exit(1)

    image_path      = sys.argv[1]
    clothing_prompt = sys.argv[2]

    print("\n" + "=" * 55)
    print("  Virtual Try-On — Inpainting Pipeline")
    print("=" * 55)
    print(f"  Image  : {image_path}")
    print(f"  Prompt : {clothing_prompt}")
    print("=" * 55)

    original, result = run_pipeline(image_path, clothing_prompt)

    print("\n" + "=" * 55)
    print(f"  Done!  Output: {result.size[0]}x{result.size[1]}px")
    print("  samples/output.png       -> result")
    print("  samples/mask_overlay.png -> mask used")
    print("  samples/comparison_*.png -> before vs after")
    print("=" * 55)
    