"""
Phase 2 — Clothing Mask Generation (Smart Item Detection)
===========================================================
Detects exactly which clothing item is in the prompt and
masks only that specific region — nothing else.

Model label reference (mattmdjaga/segformer_b2_clothes):
  Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt,
  Pants, Dress, Belt, Left-shoe, Right-shoe, Face,
  Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf

Usage:
    python segmentation.py samples/input.jpg
    python segmentation.py samples/input.jpg 15 upper
    python segmentation.py samples/input.jpg 15 lower
"""

import os
import sys
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageDraw
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# Mapping of detected clothing type → exact model labels to mask
# Only the relevant region gets masked — nothing else
CATEGORY_LABEL_MAP = {
    "upper":  ["upper-clothes"],
    "lower":  ["pants", "skirt", "left-leg", "right-leg"],
    "full":   ["upper-clothes", "pants", "skirt", "dress", "left-leg", "right-leg", "left-arm", "right-arm"],
    "dress":  ["dress", "upper-clothes", "left-leg", "right-leg"],
    "jacket": ["upper-clothes"],
    "shirt":  ["upper-clothes"],
    "hoodie": ["upper-clothes"],
    "coat":   ["upper-clothes"],
    "pants":  ["pants", "left-leg", "right-leg"],
    "jeans":  ["pants", "left-leg", "right-leg"],
    "skirt":  ["skirt", "left-leg", "right-leg"],
    "shorts": ["pants", "left-leg", "right-leg"],
    "scarf":  ["scarf"],
    "hat":    ["hat"],
}

TARGET_SIZE   = (512, 512)
MASK_DILATION = 15


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_seg_pipeline = None

def _load_model():
    global _seg_pipeline
    if _seg_pipeline is None:
        print(f"[segmentation] Loading SegFormer model...")
        from transformers import (
            SegformerImageProcessor,
            AutoModelForSemanticSegmentation,
            pipeline as hf_pipeline,
        )
        processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        model     = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
        _seg_pipeline = hf_pipeline(
            "image-segmentation",
            model=model,
            image_processor=processor,
        )
        print("[segmentation] Model loaded.")
    return _seg_pipeline


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def remove_background(image: Image.Image) -> tuple:
    try:
        from rembg import remove
        print("[segmentation] Removing background...")
        image_no_bg = remove(image)
        alpha = image_no_bg.split()[3]
        person_mask = alpha.point(lambda p: 255 if p > 127 else 0)
        print("[segmentation] Background removed.")
        return image_no_bg, person_mask
    except ImportError:
        print("[segmentation] rembg not found. Run: pip install 'rembg[cpu]'")
        return None, None


def image_no_bg_to_white(image_no_bg: Image.Image) -> Image.Image:
    white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
    return white_bg


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    return image


def run_segmentation(image: Image.Image) -> list:
    model = _load_model()
    segments = model(image)
    print(f"[segmentation] Detected labels: {[s['label'] for s in segments]}")
    return segments


def extract_mask_for_labels(segments: list, target_labels: list) -> Image.Image | None:
    """
    Build a binary mask from only the specified labels.
    Everything else is left black (untouched).
    """
    combined_mask = None
    matched = []

    for seg in segments:
        if seg["label"].lower() in [l.lower() for l in target_labels]:
            matched.append(seg["label"])
            seg_mask = seg["mask"].convert("L")
            seg_mask = seg_mask.point(lambda p: 255 if p > 0 else 0)
            if combined_mask is None:
                combined_mask = seg_mask
            else:
                combined_mask = ImageChops.lighter(combined_mask, seg_mask)

    if combined_mask is None:
        print(f"[segmentation] WARNING: None of {target_labels} found in image.")
    else:
        print(f"[segmentation] Matched: {matched} — coverage: {_mask_coverage(combined_mask):.1f}%")

    return combined_mask


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------

def refine_with_silhouette(clothing_mask: Image.Image, person_mask: Image.Image) -> Image.Image:
    refined = ImageChops.darker(clothing_mask, person_mask)
    print(f"[segmentation] Refined: {_mask_coverage(clothing_mask):.1f}% → {_mask_coverage(refined):.1f}%")
    return refined


def dilate_mask(mask: Image.Image, size: int = MASK_DILATION) -> Image.Image:
    if size % 2 == 0:
        size += 1
    return mask.filter(ImageFilter.MaxFilter(size))


def smooth_mask_edges(mask: Image.Image, blur_radius: int = 3) -> Image.Image:
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return blurred.point(lambda p: 255 if p > 127 else 0)


def fallback_mask(image: Image.Image, category: str) -> Image.Image:
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if category == "lower":
        draw.rectangle([int(w*0.12), int(h*0.48), int(w*0.88), int(h*0.98)], fill=255)
    elif category == "full":
        draw.rectangle([int(w*0.10), int(h*0.12), int(w*0.90), int(h*0.98)], fill=255)
    else:
        draw.rectangle([int(w*0.10), int(h*0.12), int(w*0.90), int(h*0.72)], fill=255)
    print(f"[segmentation] Using fallback [{category}] rectangle mask.")
    return mask


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def get_clothing_mask(
    image_path: str,
    dilation: int = MASK_DILATION,
    category: str = "upper",
    use_fallback: bool = True,
    save_debug: bool = False,
) -> tuple:
    """
    Generate a mask for only the clothing region specified by category.

    Args:
        image_path: Path to person photo
        dilation:   Edge expansion in pixels
        category:   Clothing type key from CATEGORY_LABEL_MAP
                    e.g. "upper", "lower", "jeans", "dress", "hoodie"
        use_fallback: Use rectangle if segmentation fails

    Returns:
        (image, mask) — both at 512x512
    """
    # Resolve category to model labels
    target_labels = CATEGORY_LABEL_MAP.get(category, CATEGORY_LABEL_MAP["upper"])
    print(f"[segmentation] Item: [{category}] → masking labels: {target_labels}")

    # Load
    image = load_image(image_path)

    # Remove background
    image_no_bg, person_mask = remove_background(image)
    image_for_seg = image_no_bg_to_white(image_no_bg) if image_no_bg is not None else image

    # Segment
    segments = run_segmentation(image_for_seg)

    # Extract mask for target labels only
    mask = extract_mask_for_labels(segments, target_labels)

    # Refine with person silhouette
    if mask is not None and person_mask is not None:
        mask = refine_with_silhouette(mask, person_mask.resize(TARGET_SIZE, Image.LANCZOS))

    # Fallback
    if mask is None or _mask_coverage(mask) < 2.0:
        if use_fallback:
            # Determine broad category for fallback shape
            broad = "lower" if category in ("lower", "pants", "jeans", "skirt", "shorts") else \
                    "full"  if category in ("full", "dress") else "upper"
            mask = fallback_mask(image, broad)
            if person_mask is not None:
                mask = refine_with_silhouette(mask, person_mask.resize(TARGET_SIZE, Image.LANCZOS))
        else:
            raise ValueError(f"Could not generate mask for [{category}].")

    # Dilate and smooth
    mask = dilate_mask(mask, size=dilation)
    mask = smooth_mask_edges(mask)

    print(f"[segmentation] Final mask coverage: {_mask_coverage(mask):.1f}%")

    if save_debug:
        image.save("debug_image.png")
        mask.save("debug_mask.png")
        if image_no_bg is not None:
            image_no_bg_to_white(image_no_bg).save("debug_nobg.png")
        print("[segmentation] Saved debug images.")

    return image, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_coverage(mask: Image.Image) -> float:
    arr = np.array(mask)
    return (np.sum(arr > 127) / arr.size) * 100


def visualize_mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    overlay   = image.copy().convert("RGBA")
    red_layer = Image.new("RGBA", image.size, (220, 50, 50, int(255 * alpha)))
    overlay.paste(red_layer, mask=mask)
    return overlay.convert("RGB")


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python segmentation.py <image> [dilation] [category]")
        print("Example: python segmentation.py samples/input.jpg 15 jeans")
        print(f"Categories: {list(CATEGORY_LABEL_MAP.keys())}")
        sys.exit(1)

    image_path = sys.argv[1]
    dilation   = int(sys.argv[2])    if len(sys.argv) > 2 else MASK_DILATION
    category   = sys.argv[3].lower() if len(sys.argv) > 3 else "upper"

    print(f"\n--- Phase 2: Clothing Mask Generation ---")
    print(f"Input    : {image_path}")
    print(f"Dilation : {dilation}px")
    print(f"Category : {category}\n")

    image, mask = get_clothing_mask(
        image_path, dilation=dilation, category=category, save_debug=True,
    )

    mask.save("output_mask.png")
    visualize_mask_overlay(image, mask).save("output_overlay.png")

    print(f"\n--- Done --- coverage: {_mask_coverage(mask):.1f}%")
    print("output_mask.png    -> binary mask")
    print("output_overlay.png -> red overlay")