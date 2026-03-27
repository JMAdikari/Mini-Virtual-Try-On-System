import os
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageDraw
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# Labels from the model that we consider "clothing to replace"
CLOTHING_LABELS = [
    "upper-clothes",
    "dress",
    "coat",
    "jacket",
    "shirt",
    "sweater",
    "skirt",
    "pants",
    "shorts",
    "jumpsuit",
    "cape",
]

TARGET_SIZE = (512, 512)   # Stable Diffusion native resolution
MASK_DILATION = 15         # Pixels to expand mask edges for smoother blending


# ---------------------------------------------------------------------------
# Model loader (lazy — only loads when first called)
# ---------------------------------------------------------------------------

_seg_pipeline = None

def _load_model():
    global _seg_pipeline
    if _seg_pipeline is None:
        print(f"[segmentation] Loading model: {MODEL_ID}")
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, pipeline as hf_pipeline
        import torch

        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

        _seg_pipeline = hf_pipeline(
            "image-segmentation",
            model=model,
            image_processor=processor,
        )
        print("[segmentation] Model loaded.")
    return _seg_pipeline


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> Image.Image:
    """Load and resize image to 512x512 RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.LANCZOS)
    return image


def run_segmentation(image: Image.Image) -> list:
    """
    Run the clothing segmentation model on an image.

    Returns:
        List of dicts: [{"label": str, "score": float, "mask": PIL.Image}, ...]
    """
    model = _load_model()
    segments = model(image)
    print(f"[segmentation] Detected labels: {[s['label'] for s in segments]}")
    return segments


def extract_clothing_mask(segments: list, labels: list = CLOTHING_LABELS) -> Image.Image | None:
    """
    Merge all clothing segment masks into a single binary mask.

    White pixels (255) = areas to replace (clothing)
    Black pixels (0)   = areas to keep (face, background, etc.)

    Args:
        segments: Output from run_segmentation()
        labels:   List of label strings to treat as clothing

    Returns:
        Combined PIL mask image, or None if no clothing detected
    """
    combined_mask = None

    for seg in segments:
        if seg["label"].lower() in [l.lower() for l in labels]:  # case-insensitive match
            seg_mask = seg["mask"].convert("L")  # ensure grayscale
            # Binarize: any non-zero pixel becomes 255
            seg_mask = seg_mask.point(lambda p: 255 if p > 0 else 0)

            if combined_mask is None:
                combined_mask = seg_mask
            else:
                combined_mask = ImageChops.lighter(combined_mask, seg_mask)

    if combined_mask is None:
        print("[segmentation] WARNING: No clothing labels detected in image.")
    else:
        coverage = _mask_coverage(combined_mask)
        print(f"[segmentation] Clothing mask coverage: {coverage:.1f}%")

    return combined_mask


def dilate_mask(mask: Image.Image, size: int = MASK_DILATION) -> Image.Image:
    """
    Expand mask edges outward by `size` pixels.

    This prevents hard seams between the inpainted area and the
    original image. Larger values = smoother blend, less precision.

    Args:
        mask: Binary PIL mask
        size: Dilation kernel size (odd number recommended)

    Returns:
        Dilated PIL mask
    """
    if size % 2 == 0:
        size += 1  # MaxFilter requires odd size
    dilated = mask.filter(ImageFilter.MaxFilter(size))
    return dilated


def smooth_mask_edges(mask: Image.Image, blur_radius: int = 3) -> Image.Image:
    """
    Apply a slight Gaussian blur to mask edges for a feathered transition.
    Then re-binarize to keep it clean for inpainting.

    Args:
        mask: Binary PIL mask
        blur_radius: Blur amount (2-5 recommended)

    Returns:
        Smoothed PIL mask
    """
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Re-threshold: keep pixels above 50% as white
    smoothed = blurred.point(lambda p: 255 if p > 127 else 0)
    return smoothed


def fallback_torso_mask(image: Image.Image) -> Image.Image:
    """
    Manual fallback mask if segmentation fails completely.

    Draws a rectangle covering the approximate torso region
    (upper 25% to lower 75% of image height, centered horizontally).

    Use this only as a last resort to keep the pipeline running.

    Args:
        image: PIL image (used for size reference)

    Returns:
        Binary PIL mask with a torso-region rectangle
    """
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Torso box: x from 20%-80%, y from 20%-75%
    x0, y0 = int(w * 0.20), int(h * 0.20)
    x1, y1 = int(w * 0.80), int(h * 0.75)
    draw.rectangle([x0, y0, x1, y1], fill=255)

    print(f"[segmentation] Using fallback torso mask: ({x0},{y0}) -> ({x1},{y1})")
    return mask


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def get_clothing_mask(
    image_path: str,
    dilation: int = MASK_DILATION,
    use_fallback: bool = True,
    save_debug: bool = False,
) -> tuple[Image.Image, Image.Image]:
    """
    Full Phase 2 pipeline: load image -> segment -> extract mask -> dilate.

    Args:
        image_path:   Path to person photo
        dilation:     How much to expand mask edges (default 15px)
        use_fallback: If True, use torso rectangle if segmentation fails
        save_debug:   If True, save intermediate masks to disk for inspection

    Returns:
        (image, mask) — both PIL images at 512x512
        image: resized original
        mask:  binary clothing mask, ready for inpainting
    """
    # Step 1 — Load and resize
    image = load_image(image_path)

    # Step 2 — Segment
    segments = run_segmentation(image)

    # Step 3 — Extract clothing mask
    mask = extract_clothing_mask(segments)

    # Step 4 — Fallback if segmentation found nothing
    if mask is None:
        if use_fallback:
            print("[segmentation] Falling back to manual torso mask.")
            mask = fallback_torso_mask(image)
        else:
            raise ValueError(
                "Segmentation found no clothing in this image. "
                "Try a photo with a clear view of the upper body, "
                "or set use_fallback=True."
            )

    # Step 5 — Dilate for smoother inpainting blend
    mask = dilate_mask(mask, size=dilation)
    mask = smooth_mask_edges(mask)

    # Optional: save debug images
    if save_debug:
        image.save("debug_image.png")
        mask.save("debug_mask.png")
        print("[segmentation] Saved debug_image.png and debug_mask.png")

    return image, mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_coverage(mask: Image.Image) -> float:
    """Return percentage of mask pixels that are white (clothing area)."""
    arr = np.array(mask)
    white_pixels = np.sum(arr > 127)
    total_pixels = arr.size
    return (white_pixels / total_pixels) * 100


def visualize_mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    """
    Overlay the mask on the image in red for visual debugging.

    Args:
        image: Original RGB image
        mask:  Binary mask
        alpha: Opacity of the overlay (0.0-1.0)

    Returns:
        PIL image with red overlay showing the masked region
    """
    overlay = image.copy().convert("RGBA")
    red_layer = Image.new("RGBA", image.size, (220, 50, 50, int(255 * alpha)))
    mask_rgba = mask.convert("RGBA")

    # Apply red only where mask is white
    composite = Image.composite(red_layer, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_rgba)
    overlay = Image.alpha_composite(overlay, composite)
    return overlay.convert("RGB")


# ---------------------------------------------------------------------------
# Run standalone for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <image_path> [dilation_size]")
        print("Example: python segmentation.py samples/input.jpg 15")
        sys.exit(1)

    image_path = sys.argv[1]
    dilation = int(sys.argv[2]) if len(sys.argv) > 2 else MASK_DILATION

    print(f"\n--- Running Phase 2: Clothing Mask Generation ---")
    print(f"Input: {image_path}")
    print(f"Dilation: {dilation}px\n")

    image, mask = get_clothing_mask(
        image_path,
        dilation=dilation,
        save_debug=True,
    )

    # Save outputs
    mask.save("output_mask.png")
    overlay = visualize_mask_overlay(image, mask)
    overlay.save("output_overlay.png")

    print("\n--- Done ---")
    print("output_mask.png    -> binary mask (white = clothing area)")
    print("output_overlay.png -> red overlay showing detected region")
    print("debug_image.png    -> resized input image")