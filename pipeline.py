"""
Phase 3 - Inpainting Pipeline (Replicate API)
=============================================
Smart clothing detection from prompt.
Color accuracy fixes: guidance 12.0, prompt repeated 3x, wrong colors blocked.

Usage:
    python pipeline.py samples/input.jpg "blue oversized hoodie"
    python pipeline.py samples/input.jpg "black skinny jeans"
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

from segmentation import get_clothing_mask, visualize_mask_overlay, CATEGORY_LABEL_MAP

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
OUTPUT_DIR = "samples"

# ---------------------------------------------------------------------------
# Item detection
# ---------------------------------------------------------------------------

ITEM_KEYWORD_MAP = [
    ("jumpsuit", "full"),
    ("romper", "full"),
    ("overalls", "full"),
    ("bodysuit", "full"),
    ("full outfit", "full"),
    ("outfit", "full"),
    ("suit", "full"),
    ("sundress", "dress"),
    ("mini dress", "dress"),
    ("maxi dress", "dress"),
    ("dress", "dress"),
    ("gown", "dress"),
    ("jeans", "jeans"),
    ("pants", "pants"),
    ("trousers", "pants"),
    ("chinos", "pants"),
    ("leggings", "pants"),
    ("joggers", "pants"),
    ("sweatpants", "pants"),
    ("cargo pants", "pants"),
    ("flare pants", "pants"),
    ("slacks", "pants"),
    ("khakis", "pants"),
    ("shorts", "shorts"),
    ("skirt", "skirt"),
    ("culottes", "skirt"),
    ("hoodie", "hoodie"),
    ("sweatshirt", "shirt"),
    ("jacket", "jacket"),
    ("blazer", "jacket"),
    ("coat", "coat"),
    ("parka", "coat"),
    ("shirt", "shirt"),
    ("tee", "shirt"),
    ("t-shirt", "shirt"),
    ("top", "shirt"),
    ("blouse", "shirt"),
    ("sweater", "shirt"),
    ("jumper", "shirt"),
    ("turtleneck", "shirt"),
    ("cardigan", "shirt"),
    ("vest", "shirt"),
    ("crop top", "shirt"),
    ("scarf", "scarf"),
    ("hat", "hat"),
    ("cap", "hat"),
    ("beanie", "hat"),
]

STRENGTH_MAP = {
    "upper": 0.55,
    "lower": 0.75,
    "full": 0.75,
    "dress": 0.75,
    "jacket": 0.60,
    "shirt": 0.60,
    "hoodie": 0.65,
    "coat": 0.60,
    "pants": 0.75,
    "jeans": 0.75,
    "skirt": 0.75,
    "shorts": 0.75,
    "scarf": 0.50,
    "hat": 0.50,
}

GUIDANCE_MAP = {
    "upper": 12.0,
    "lower": 11.0,
    "full": 11.0,
    "dress": 11.0,
    "jacket": 12.0,
    "shirt": 12.0,
    "hoodie": 12.0,
    "coat": 12.0,
    "pants": 11.0,
    "jeans": 11.0,
    "skirt": 11.0,
    "shorts": 11.0,
    "scarf": 10.0,
    "hat": 10.0,
}

LOCK_MAP = {
    "upper": "same pants, same lower body, same shoes",
    "lower": "same upper body, same shirt, same jacket, same top",
    "full": "same face, same hairstyle, same shoes",
    "dress": "same face, same hairstyle, same shoes",
    "jacket": "same pants, same lower body, same shirt underneath",
    "shirt": "same pants, same lower body, same jacket if any",
    "hoodie": "same pants, same lower body, same shoes",
    "coat": "same pants, same lower body, same clothes underneath",
    "pants": "same upper body, same shirt, same jacket, same shoes",
    "jeans": "same upper body, same shirt, same jacket, same shoes",
    "skirt": "same upper body, same shirt, same jacket, same shoes",
    "shorts": "same upper body, same shirt, same jacket, same shoes",
    "scarf": "same clothing, same face, same body",
    "hat": "same clothing, same face, same body",
}


def detect_item(prompt):
    p = prompt.lower()
    for keyword, category in ITEM_KEYWORD_MAP:
        if keyword in p:
            print("[pipeline] Detected item: '{}' -> [{}]".format(keyword, category))
            return category
    print("[pipeline] No specific item detected -> [upper]")
    return "upper"


def extract_wrong_colors(prompt):
    prompt_lower = prompt.lower()
    priority_block = ["black", "dark", "maroon", "burgundy", "brown", "grey", "gray", "dark red"]
    wrong = [c for c in priority_block if c not in prompt_lower]
    return ", ".join(wrong[:5])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(image, fmt="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return "data:{};base64,{}".format(mime, b64)


def _download_image(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def _save_comparison(original, result, prompt):
    w, h = original.size
    comparison = Image.new("RGB", (w * 2 + 20, h), (240, 240, 240))
    comparison.paste(original, (0, 0))
    comparison.paste(result, (w + 20, 0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = prompt[:40].replace(" ", "_").replace("/", "-")
    path = os.path.join(OUTPUT_DIR, "comparison_{}.png".format(safe))
    comparison.save(path)
    return path

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(clothing_prompt, item_category):
    lock = LOCK_MAP.get(item_category, "same lower body, same face, same pose")
    is_lower = item_category in ("lower", "pants", "jeans", "skirt", "shorts")
    is_full = item_category in ("full", "dress")
    wrong_colors = extract_wrong_colors(clothing_prompt)

    # Repeat 3x for stronger color signal
    prompt = (
        "person wearing {}, "
        "{}, "
        "clothing color and style: {}, "
        "{}, "
        "same face, same background, same pose, same body position, "
        "photorealistic, high quality, sharp focus, natural fabric texture"
    ).format(clothing_prompt, clothing_prompt, clothing_prompt, lock)

    negative_parts = [
        "{} clothing".format(wrong_colors),
        "necklace, jewelry, accessories, chains, pendants, rings, bracelets, earrings",
        "different pose, crossed arms, changed body position, changed hands",
        "different person, different face, different hair, changed background",
        "cartoon, anime, blurry, low quality, deformed, extra limbs, bad anatomy",
        "watermark, text, logo, duplicate",
    ]

    if is_lower:
        negative_parts.insert(0, "dark jeans, navy jeans, blue jeans, dark pants")
        negative_parts.append("different upper body, changed shirt, changed jacket, changed top")

    if not is_lower and not is_full:
        negative_parts.append("different pants, changed lower body, different shoes")

    negative_prompt = ", ".join(negative_parts)
    return prompt, negative_prompt

# ---------------------------------------------------------------------------
# Replicate API
# ---------------------------------------------------------------------------

def run_inpainting(image, mask, prompt, negative_prompt, original_size, strength=0.60, guidance_scale=12.0, num_inference_steps=50):
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not found. Add to .env file.")

    print("[pipeline] Converting images to base64...")
    image_b64 = _pil_to_base64(image)
    mask_b64 = _pil_to_base64(mask)

    headers = {
        "Authorization": "Token {}".format(REPLICATE_API_TOKEN),
        "Content-Type": "application/json",
    }

    payload = {
        "version": REPLICATE_MODEL.split(":")[1],
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image_b64,
            "mask": mask_b64,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "width": 512,
            "height": 512,
            "num_outputs": 1,
        }
    }

    print("[pipeline] Strength: {} | Guidance: {}".format(strength, guidance_scale))
    print("[pipeline] Sending to Replicate API...")

    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers, json=payload, timeout=30,
    )
    if response.status_code != 201:
        raise RuntimeError("Replicate API error {}: {}".format(response.status_code, response.text))

    prediction_id = response.json()["id"]
    print("[pipeline] Prediction ID: {}".format(prediction_id))
    print("[pipeline] Waiting for result (30-60 seconds)...")

    poll_url = "https://api.replicate.com/v1/predictions/{}".format(prediction_id)
    while True:
        poll = requests.get(poll_url, headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data["status"]
        if status == "succeeded":
            print("[pipeline] Generation complete!")
            output_url = data["output"][0]
            break
        elif status == "failed":
            raise RuntimeError("Prediction failed: {}".format(data.get("error", "Unknown")))
        else:
            print("[pipeline] Status: {}... waiting".format(status))
            time.sleep(3)

    print("[pipeline] Downloading result...")
    result = _download_image(output_url)

    if result.size != original_size:
        print("[pipeline] Resizing {} -> {}".format(result.size, original_size))
        result = result.resize(original_size, Image.LANCZOS)

    return result

# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def run_pipeline(image_path, clothing_prompt):
    original_size = Image.open(image_path).convert("RGB").size
    print("[pipeline] Original size: {}x{}px".format(original_size[0], original_size[1]))

    print("\n[pipeline] Step 1: Detecting clothing item...")
    item_category = detect_item(clothing_prompt)
    strength = STRENGTH_MAP.get(item_category, 0.60)
    guidance_scale = GUIDANCE_MAP.get(item_category, 12.0)
    wrong_colors = extract_wrong_colors(clothing_prompt)
    print("[pipeline] Strength: {} | Guidance: {}".format(strength, guidance_scale))
    print("[pipeline] Blocking wrong colors: {}".format(wrong_colors))

    print("\n[pipeline] Step 2: Generating mask for [{}]...".format(item_category))
    image, mask = get_clothing_mask(image_path, category=item_category, save_debug=False)

    prompt, negative_prompt = build_prompt(clothing_prompt, item_category)
    print("\n[pipeline] Step 3: Prompt ready")
    print("  Prompt  : {}".format(prompt))
    print("  Negative: {}".format(negative_prompt))

    print("\n[pipeline] Step 4: Running inpainting...")
    result = run_inpainting(
        image, mask, prompt, negative_prompt,
        original_size, strength=strength, guidance_scale=guidance_scale,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_original = image.resize(original_size, Image.LANCZOS)

    result.save(os.path.join(OUTPUT_DIR, "output.png"))
    print("\n[pipeline] Saved: samples/output.png ({}x{}px)".format(result.size[0], result.size[1]))

    comp = _save_comparison(image_original, result, clothing_prompt)
    print("[pipeline] Saved: {}".format(comp))

    overlay = visualize_mask_overlay(image, mask)
    overlay.resize(original_size, Image.LANCZOS).save(os.path.join(OUTPUT_DIR, "mask_overlay.png"))
    print("[pipeline] Saved: samples/mask_overlay.png")

    return image_original, result

# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:   python pipeline.py <image> \"<prompt>\"")
        print("Examples:")
        print('  python pipeline.py samples/input.jpg "blue oversized hoodie"')
        print('  python pipeline.py samples/input.jpg "black skinny jeans"')
        print('  python pipeline.py samples/input.jpg "floral summer dress"')
        sys.exit(1)

    image_path = sys.argv[1]
    clothing_prompt = sys.argv[2]

    print("\n" + "=" * 55)
    print("  Virtual Try-On - Smart Clothing Detection")
    print("=" * 55)
    print("  Image  : {}".format(image_path))
    print("  Prompt : {}".format(clothing_prompt))
    print("=" * 55)

    original, result = run_pipeline(image_path, clothing_prompt)

    print("\n" + "=" * 55)
    print("  Done! Output: {}x{}px".format(result.size[0], result.size[1]))
    print("  samples/output.png       -> result")
    print("  samples/mask_overlay.png -> what was masked")
    print("  samples/comparison_*.png -> before vs after")
    print("=" * 55)