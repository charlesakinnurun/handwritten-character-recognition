"""Interactive Streamlit demo for Handwriting Intelligence.

Run with::

    streamlit run app/streamlit_app.py

The demo loads a trained **inference artifact** and classifies a single handwritten
character, either uploaded or drawn on a canvas. It never claims word/sentence
recognition: the loaded models classify isolated characters only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from handwriting_intelligence.inference.predictor import Predictor, load_predictor

st.set_page_config(page_title="Handwriting Intelligence", page_icon="✍️", layout="wide")

DEFAULT_ARTIFACT = os.environ.get("HWI_ARTIFACT_PATH", "models/mnist_cnn/artifact.pt")


@st.cache_resource(show_spinner="Loading model artifact...")
def _load(artifact_path: str) -> Predictor:
    return load_predictor(artifact_path)


def _sidebar() -> tuple[str, int]:
    st.sidebar.header("Model")
    artifact_path = st.sidebar.text_input("Artifact path", value=DEFAULT_ARTIFACT)
    top_k = st.sidebar.slider("Top-k predictions", min_value=1, max_value=10, value=5)
    st.sidebar.caption("Set `HWI_ARTIFACT_PATH` or train a model first.")
    return artifact_path, top_k


def _render_prediction(predictor: Predictor, image: Image.Image, top_k: int) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Your input")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("What the model sees")
        from handwriting_intelligence.inference.preprocessing import preprocess_display_array

        canvas = preprocess_display_array(image, predictor.preprocessing)
        st.image(canvas, clamp=True, use_container_width=True)

    prediction = predictor.predict(image, top_k=top_k)
    st.success(f"Prediction: **{prediction.label}**  (confidence {prediction.confidence:.1%})")
    st.caption(
        "Confidence is the model's softmax probability for the top class - it is a "
        "statistical score, not a guarantee that the character is correct."
    )
    st.write("Top predictions")
    for alt in prediction.top_k:
        st.write(f"`{alt.label}`")
        st.progress(min(1.0, alt.probability), text=f"{alt.probability:.2%}")
    st.caption(f"model: {prediction.model_name} v{prediction.model_version}")


def _draw_tab(predictor: Predictor, top_k: int) -> None:
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.info(
            "Drawing requires the optional dependency `streamlit-drawable-canvas`. "
            'Install it with `pip install streamlit-drawable-canvas` (or `pip install -e ".[demo]"`), '
            "or use the Upload tab."
        )
        return

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None and canvas_result.image_data[..., :3].max() > 0:
        rgba = canvas_result.image_data.astype(np.uint8)
        image = Image.fromarray(rgba[..., :3]).convert("L")
        _render_prediction(predictor, image, top_k)
    else:
        st.info("Draw a single character on the canvas (white stroke on black).")


def main() -> None:
    st.title("✍️ Handwriting Intelligence")
    st.write(
        "Recognise a single handwritten **digit or character** from an image. "
        "This demo classifies **isolated characters** only."
    )

    artifact_path, top_k = _sidebar()
    if not Path(artifact_path).is_file():
        st.error(
            f"No artifact found at `{artifact_path}`. Train one with "
            "`python -m handwriting_intelligence.cli.train --config configs/mnist_cnn.yaml`."
        )
        return

    predictor = _load(artifact_path)
    info = predictor.info()
    st.info(
        f"Loaded **{info['model_name']}** ({info['class_kind']}, {info['num_classes']} classes) "
        f"- {info['class_description']}\n\n"
        "Word/sentence recognition is **not** supported by this model; see the CRNN "
        "extension in `docs/architecture.md`."
    )

    with st.expander("Model information"):
        st.json(info)

    upload_tab, draw_tab = st.tabs(["Upload an image", "Draw a character"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a character image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("L")
            _render_prediction(predictor, image, top_k)
        else:
            st.caption("Upload a single character image (PNG/JPEG).")
    with draw_tab:
        _draw_tab(predictor, top_k)


main()
