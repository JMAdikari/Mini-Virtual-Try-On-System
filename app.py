"""
Phase 4 — Gradio UI
====================
Simple web interface for the Mini Virtual Try-On System.
Lets anyone upload a photo and type a clothing prompt to
see a before/after result without touching the terminal.

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import os
import gradio as gr
from PIL import Image
from pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Core function wired to Gradio
# ---------------------------------------------------------------------------

def try_on(image_path: str, clothing_prompt: str):
    """
    Called by Gradio when the user clicks Generate.

    Args:
        image_path:      Temp path Gradio saves the uploaded image to
        clothing_prompt: Text entered by the user

    Returns:
        (original, result) PIL images shown side by side in the UI
    """
    if not image_path:
        raise gr.Error("Please upload a photo first.")
    if not clothing_prompt or clothing_prompt.strip() == "":
        raise gr.Error("Please enter a clothing description.")

    original, result = run_pipeline(image_path, clothing_prompt.strip())
    return original, result


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Mini Virtual Try-On") as demo:

    gr.Markdown("# Mini Virtual Try-On System")
    gr.Markdown(
        "Upload a photo of a person and describe the clothing you want them to wear. "
        "The AI will replace their existing outfit with your description."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="filepath",
                label="Upload person photo",
            )
            prompt_input = gr.Textbox(
                label="Clothing description",
                placeholder='e.g. black oversized hoodie',
                lines=2,
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            before_output = gr.Image(label="Before")
            after_output  = gr.Image(label="After")

    gr.Markdown("### Example prompts")
    gr.Examples(
        examples=[
            ["samples/input.jpg", "black oversized hoodie"],
            ["samples/input.jpg", "red floral summer dress"],
            ["samples/input.jpg", "white linen button-up shirt"],
            ["samples/input.jpg", "navy blue blazer"],
        ],
        inputs=[image_input, prompt_input],
    )

    generate_btn.click(
        fn=try_on,
        inputs=[image_input, prompt_input],
        outputs=[before_output, after_output],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()