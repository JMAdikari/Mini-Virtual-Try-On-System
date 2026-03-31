"""
Phase 2 — Clothing Mask Generation (Production Level)
=======================================================
Strategy:
  1. Remove background with rembg (eliminates color confusion)
  2. Run SegFormer on the clean person-only image
  3. Extract clothing mask based on detected category
  4. Combine with person silhouette mask for best coverage
  5. Dilate + smooth edges for clean inpainting blend

Supports upper body, lower body, and full body clothing.

Model label reference (mattmdjaga/segformer_b2_clothes):
  Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt,
  Pants, Dress, Belt, Left-shoe, Right-shoe, Face,
  Left-leg, Right-leg, Left-arm, Right-arm, Bag, Scarf

Usage:
    python segmentation.py samples/input.jpg
    python segmentation.py samples/input.jpg 15 lower
    python segmentation.py samples/input.jpg 15 full
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

# Exact labels returned by mattmdjaga/segformer_b2_clothes (lowercased)
# Upper body — shirt/jacket/coat all fall under "upper-clothes"
UPPER_BODY_LABELS = [
    "upper-clothes",
    "scarf",
]

# Lower body — pants + leg segments for full coverage
LOWER_BODY_LABELS = [
    "pants",
    "skirt",
    "left-leg",
    "right-leg",
    "belt",
]

# Full body — everything clothing-related
FULL_BODY_LABELS = [
    "upper-clothes",
    "pants",
    "skirt",
    "dress",
    "left-leg",
    "right-leg",
    "left-arm",
    "right-arm",
    "belt",
    "scarf",
]

TARGET_SIZE   = (512, 512)
MASK_DILATION = 15


# ---------------------------------------------------------------------------
# Model loader (lazy)
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
# Step 1 — Background removal
# ---------------------------------------------------------------------------

def remove_background(image: Image.Image) -> tuple:
    """Remove background using rembg. Returns (image_no_bg, person_mask)."""
    try:
        from rembg import remove
        print("[segmentation] Removing background with rembg...")
        image_no_bg = remove(image)
        alpha = image_no_bg.split()[3]
        person_mask = alpha.point(lambda p: 255 if p > 127 else 0)
        print("[segmentation] Background removed.")
        return image_no_bg, person_mask
    except ImportError:
        print("[segmentation] rembg not found. Run: pip install 'rembg[cpu]'")
        return None, None


def image_no_bg_to_white(image_no_bg: Image.Image) -> Image.Image:
    """Paste person onto white background for cleaner segmentation."""
    white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
    return white_bg


# ---------------------------------------------------------------------------
# Step 2 — Segmentation
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> Image.Image:
    """Load and resize image to 512x512 RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    return image


def run_segmentation(image: Image.Image) -> list:
    """Run SegFormer on an image. Returns list of segment dicts."""
    model = _load_model()
    segments = model(image)
    print(f"[segmentation] Detected labels: {[s['label'] for s in segments]}")
    return segments


def extract_clothing_mask(segments: list, category: str = "upper") -> Image.Image | None:
    """
    Merge all clothing segment masks into one binary mask.

    Args:
        segments: Output from run_segmentation()
        category: "upper", "lower", or "full"

    Returns:
        Combined binary mask or None
    """
    if category == "lower":
        target_labels = LOWER_BODY_LABELS
    elif category == "full":
        target_labels = FULL_BODY_LABELS
    else:
        target_labels = UPPER_BODY_LABELS

    combined_mask = None
    matched = []

    for seg in segments:
        if seg["label"].lower() in target_labels:
            matched.append(seg["label"])
            seg_mask = seg["mask"].convert("L")
            seg_mask = seg_mask.point(lambda p: 255 if p > 0 else 0)
            if combined_mask is None:
                combined_mask = seg_mask
            else:
                combined_mask = ImageChops.lighter(combined_mask, seg_mask)

    if combined_mask is None:
        print(f"[segmentation] WARNING: No [{category}] labels matched.")
        print(f"[segmentation] Expected one of: {target_labels}")
    else:
        print(f"[segmentation] Matched labels: {matched}")
        print(f"[segmentation] [{category}] mask coverage: {_mask_coverage(combined_mask):.1f}%")

    return combined_mask


# ---------------------------------------------------------------------------
# Step 3 — Mask refinement
# ---------------------------------------------------------------------------

def refine_mask_with_person_silhouette(
    clothing_mask: Image.Image,
    person_mask: Image.Image,
) -> Image.Image:
    """Intersect clothing mask with person silhouette to remove bg bleed."""
    refined = ImageChops.darker(clothing_mask, person_mask)
    print(f"[segmentation] Refined: {_mask_coverage(clothing_mask):.1f}% → {_mask_coverage(refined):.1f}%")
    return refined


def dilate_mask(mask: Image.Image, size: int = MASK_DILATION) -> Image.Image:
    """Expand mask edges outward for smoother inpainting blend."""
    if size % 2 == 0:
        size += 1
    return mask.filter(ImageFilter.MaxFilter(size))


def smooth_mask_edges(mask: Image.Image, blur_radius: int = 3) -> Image.Image:
    """Feather mask edges then re-binarize."""
    blurred  = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    smoothed = blurred.point(lambda p: 255 if p > 127 else 0)
    return smoothed


def fallback_torso_mask(image: Image.Image, category: str = "upper") -> Image.Image:
    """
    Manual rectangle fallback when segmentation fails.
    Adjusts region based on category.
    """
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if category == "lower":
        x0, y0 = int(w * 0.15), int(h * 0.48)
        x1, y1 = int(w * 0.85), int(h * 0.98)
    elif category == "full":
        x0, y0 = int(w * 0.12), int(h * 0.15)
        x1, y1 = int(w * 0.88), int(h * 0.98)
    else:  # upper
        x0, y0 = int(w * 0.12), int(h * 0.15)
        x1, y1 = int(w * 0.88), int(h * 0.72)

    draw.rectangle([x0, y0, x1, y1], fill=255)
    print(f"[segmentation] Fallback [{category}] mask: ({x0},{y0}) -> ({x1},{y1})")
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
    Full Phase 2 pipeline:
      load → remove background → segment → extract mask →
      refine with silhouette → dilate → smooth

    Args:
        image_path:   Path to person photo
        dilation:     Mask edge expansion in pixels (default 15)
        category:     "upper", "lower", or "full"
        use_fallback: Use rectangle if everything fails
        save_debug:   Save intermediate images for inspection

    Returns:
        (image, mask) — both PIL images at 512x512
    """
    print(f"[segmentation] Category: [{category}]")

    # Step 1 — Load
    image = load_image(image_path)

    # Step 2 — Remove background
    image_no_bg, person_mask = remove_background(image)
    image_for_seg = image_no_bg_to_white(image_no_bg) if image_no_bg is not None else image

    # Step 3 — Segment
    segments = run_segmentation(image_for_seg)

    # Step 4 — Extract mask for detected category
    mask = extract_clothing_mask(segments, category=category)

    # Step 5 — Refine with person silhouette
    if mask is not None and person_mask is not None:
        person_mask_resized = person_mask.resize(TARGET_SIZE, Image.LANCZOS)
        mask = refine_mask_with_person_silhouette(mask, person_mask_resized)

    # Step 6 — Fallback if mask too small or missing
    if mask is None or _mask_coverage(mask) < 2.0:
        if use_fallback:
            print(f"[segmentation] Mask too small — using [{category}] fallback rectangle.")
            mask = fallback_torso_mask(image, category=category)
            if person_mask is not None:
                person_mask_resized = person_mask.resize(TARGET_SIZE, Image.LANCZOS)
                mask = refine_mask_with_person_silhouette(mask, person_mask_resized)
        else:
            raise ValueError("Could not generate a valid clothing mask.")

    # Step 7 — Dilate and smooth
    mask = dilate_mask(mask, size=dilation)
    mask = smooth_mask_edges(mask)

    print(f"[segmentation] Final mask coverage: {_mask_coverage(mask):.1f}%")

    # Debug
    if save_debug:
        image.save("debug_image.png")
        mask.save("debug_mask.png")
        if image_no_bg is not None:
            image_no_bg_to_white(image_no_bg).save("debug_nobg.png")
        print("[segmentation] Saved: debug_image.png, debug_mask.png, debug_nobg.png")

    return image, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_coverage(mask: Image.Image) -> float:
    arr = np.array(mask)
    return (np.sum(arr > 127) / arr.size) * 100


def visualize_mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Overlay mask as red tint on image for visual debugging."""
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
        print("Example: python segmentation.py samples/input.jpg 15 lower")
        print("Categories: upper | lower | full")
        sys.exit(1)

    image_path = sys.argv[1]
    dilation   = int(sys.argv[2])    if len(sys.argv) > 2 else MASK_DILATION
    category   = sys.argv[3].lower() if len(sys.argv) > 3 else "upper"

    print(f"\n--- Phase 2: Clothing Mask Generation ---")
    print(f"Input    : {image_path}")
    print(f"Dilation : {dilation}px")
    print(f"Category : {category}\n")

    image, mask = get_clothing_mask(
        image_path,
        dilation=dilation,
        category=category,
        save_debug=True,
    )

    mask.save("output_mask.png")
    overlay = visualize_mask_overlay(image, mask)
    overlay.save("output_overlay.png")

    print(f"\n--- Done --- (final coverage: {_mask_coverage(mask):.1f}%)")
    print("output_mask.png    -> binary mask")
    print("output_overlay.png -> red overlay")
    print("debug_nobg.png     -> background removed version")