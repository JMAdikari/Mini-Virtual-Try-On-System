"""
Phase 4 — Gradio UI
====================
3-column photo layout: Input | Before | After all in same row.
Controls sit below the photos.

Usage:
    python app.py
Open http://localhost:7860
"""

import gradio as gr
from pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def try_on(image_path: str, clothing_prompt: str):
    if not image_path:
        raise gr.Error("Please upload a photo first.")
    if not clothing_prompt or clothing_prompt.strip() == "":
        raise gr.Error("Please enter a clothing description.")
    original, result = run_pipeline(image_path, clothing_prompt.strip())
    return original, result


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #f8f7f4 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1c1c1c !important;
}
footer, .built-with { display: none !important; }

/* ── Page header ── */
#page-header {
    padding: 28px 0 20px;
    border-bottom: 1px solid #e8e3de;
    margin-bottom: 28px;
}
#page-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1c1c1c;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}
#page-subtitle {
    font-size: 0.84rem;
    color: #aaa;
    font-weight: 300;
    line-height: 1.5;
}

/* ── Photo column labels ── */
.col-label {
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #bdb6ae;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.col-label .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.badge-input  { background: #f0ece8; color: #aaa; border: 1px solid #e4dfd9; }
.badge-before { background: #f0ece8; color: #aaa; border: 1px solid #e4dfd9; }
.badge-after  { background: #fdf4ee; color: #c0541c; border: 1px solid #f0d4c4; }

/* ── Photo panels ── */
.photo-panel {
    background: #fff !important;
    border: 1.5px solid #e4dfd9 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.gradio-image, .svelte-1ipelgc {
    border: none !important;
    border-radius: 0 !important;
    background: #fff !important;
}

/* ── Divider ── */
#controls-divider {
    border: none;
    border-top: 1px solid #e8e3de;
    margin: 24px 0 20px;
}

/* ── Controls row ── */
#controls-row {
    display: flex;
    gap: 16px;
    align-items: flex-end;
}

/* ── Textbox ── */
textarea {
    background: #fff !important;
    border: 1.5px solid #e4dfd9 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #1c1c1c !important;
    padding: 12px 14px !important;
    resize: none !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
textarea:focus {
    border-color: #c0541c !important;
    box-shadow: 0 0 0 3px rgba(192,84,28,0.08) !important;
    outline: none !important;
}
textarea::placeholder { color: #ccc !important; }

/* ── Labels ── */
label > span {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    color: #b0a8a0 !important;
}

/* ── Generate button ── */
button.primary {
    background: #c0541c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #fff !important;
    height: 46px !important;
    min-width: 140px !important;
    transition: background 0.15s, transform 0.1s !important;
    white-space: nowrap !important;
}
button.primary:hover {
    background: #a84618 !important;
    transform: translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* ── Hint chips ── */
#hint-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}
.hint-chip {
    background: #fff;
    border: 1.5px solid #e8e3de;
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 0.73rem;
    color: #999;
}
.hint-chip b { color: #666; font-weight: 500; }

/* ── Status note ── */
#status-note {
    font-size: 0.72rem;
    color: #ccc;
    margin-top: 6px;
    letter-spacing: 0.03em;
}

/* ── Example prompts ── */
#prompts-section {
    margin-top: 28px;
    padding-top: 22px;
    border-top: 1px solid #e8e3de;
}
#prompts-heading {
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #bdb6ae;
    margin-bottom: 12px;
}
.pbtn button {
    background: #fff !important;
    border: 1.5px solid #e8e3de !important;
    border-radius: 6px !important;
    color: #777 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 9px 12px !important;
    width: 100% !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: all 0.14s !important;
}
.pbtn button:hover {
    border-color: #c0541c !important;
    color: #c0541c !important;
    background: #fdf4ee !important;
}

/* Block resets */
.gradio-container .block,
.gradio-container .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=css, title="Mini Virtual Try-On") as demo:

    # Header
    gr.HTML("""
    <div id="page-header">
        <div id="page-title">Mini Virtual Try-On System</div>
        <div id="page-subtitle">
            Upload a photo · Describe the clothing · See the result — face, pose and background stay the same.
        </div>
    </div>
    """)

    # ── 3 photos in one row ──────────────────────────────────────
    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.HTML('<div class="col-label"><span class="badge badge-input">Input</span> Your photo</div>')
            image_input = gr.Image(
                type="filepath",
                label="Upload person photo",
                height=420,
                elem_classes="photo-panel",
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="col-label"><span class="badge badge-before">Before</span> Original</div>')
            before_output = gr.Image(
                label="Before",
                height=420,
                show_label=False,
                elem_classes="photo-panel",
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="col-label"><span class="badge badge-after">After</span> Generated</div>')
            after_output = gr.Image(
                label="After",
                height=420,
                show_label=False,
                elem_classes="photo-panel",
            )

    # ── Controls below photos ────────────────────────────────────
    gr.HTML('<hr id="controls-divider">')

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Clothing description",
                placeholder="e.g. black oversized hoodie",
                lines=1,
            )
        with gr.Column(scale=1, min_width=140):
            generate_btn = gr.Button("Generate →", variant="primary")

    gr.HTML("""
    <div id="hint-bar">
        <span class="hint-chip"><b>Upper</b> — hoodies, shirts, jackets</span>
        <span class="hint-chip"><b>Lower</b> — jeans, pants, skirts</span>
        <span class="hint-chip"><b>Full</b> — dresses, jumpsuits</span>
    </div>
    <div id="status-note">Category is auto-detected from your prompt &nbsp;·&nbsp; Generation takes ~30–60 sec</div>
    """)

    # ── Example prompts ──────────────────────────────────────────
    gr.HTML("""
    <div id="prompts-section">
        <div id="prompts-heading">Example prompts — click to fill</div>
    </div>
    """)

    with gr.Row():
        b1 = gr.Button("👕  Black oversized hoodie",  elem_classes="pbtn")
        b2 = gr.Button("👗  Red floral summer dress",  elem_classes="pbtn")
        b3 = gr.Button("👔  White linen button-up",    elem_classes="pbtn")
        b4 = gr.Button("🧥  Navy blue blazer",         elem_classes="pbtn")

    with gr.Row():
        b5 = gr.Button("👖  Black skinny jeans",       elem_classes="pbtn")
        b6 = gr.Button("🧤  Brown leather jacket",     elem_classes="pbtn")
        b7 = gr.Button("🩱  Grey turtleneck sweater",  elem_classes="pbtn")
        b8 = gr.Button("🧣  Olive cargo jacket",       elem_classes="pbtn")

    b1.click(lambda: "black oversized hoodie",    outputs=prompt_input)
    b2.click(lambda: "red floral summer dress",   outputs=prompt_input)
    b3.click(lambda: "white linen button-up",     outputs=prompt_input)
    b4.click(lambda: "navy blue blazer",          outputs=prompt_input)
    b5.click(lambda: "black skinny jeans",        outputs=prompt_input)
    b6.click(lambda: "brown leather jacket",      outputs=prompt_input)
    b7.click(lambda: "grey turtleneck sweater",   outputs=prompt_input)
    b8.click(lambda: "olive cargo jacket",        outputs=prompt_input)

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