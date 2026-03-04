"""
Week 3 – Streamlit app for Sign Language MNIST prediction.

Upload a hand-sign image (PNG/JPG) and the app predicts
the letter (A–Y, excluding J and Z) with a confidence score.

Run:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mlp.model import SignLanguageMLP_v2, INPUT_SIZE, NUM_CLASSES
from src.labels import LABELS, label_to_letter

MODEL_PATH = REPO_ROOT / "app" / "best_mlp_model.pt"

# ── Helpers ──────────────────────────────────────────────────────────


@st.cache_resource
def load_model():
    """Load the best MLP from the saved .pt checkpoint."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Download `best_mlp_model.pt` from Kaggle (Output tab) "
            "and place it in the `app/` folder."
        )
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = SignLanguageMLP_v2(
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accuracy = checkpoint.get("best_test_accuracy", None)
    return model, config, accuracy


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize to 28x28 grayscale, normalize to [0,1], flatten to (1, 784)."""
    img = image.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).reshape(1, INPUT_SIZE)
    return tensor


def predict(model, tensor):
    """Run inference; return predicted label index, letter, and probabilities."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
    pred_idx = int(probs.argmax())
    return pred_idx, label_to_letter(pred_idx), probs


# ── Streamlit UI ─────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Sign Language Translator",
        page_icon="🤟",
        layout="centered",
    )

    st.title("Sign Language MNIST – Letter Predictor")
    st.markdown(
        "Upload a hand-sign image and the model predicts the letter "
        "**(A–Y, excluding J & Z)**."
    )

    model, config, train_acc = load_model()

    # Sidebar: model info
    with st.sidebar:
        st.header("Model info")
        st.write(f"**Architecture:** MLP v2 (BatchNorm)")
        st.write(f"**Hidden layers:** {config['hidden_sizes']}")
        st.write(f"**Activation:** {config['activation']}")
        st.write(f"**Dropout:** {config['dropout']}")
        if train_acc is not None:
            st.write(f"**Test accuracy:** {train_acc * 100:.2f}%")
        st.divider()
        

    uploaded = st.file_uploader(
        "Upload a hand-sign image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    if uploaded is not None:
        image = Image.open(uploaded)

        col_img, col_proc = st.columns(2)
        with col_img:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col_proc:
            st.subheader("Preprocessed (28×28)")
            gray28 = image.convert("L").resize((28, 28), Image.LANCZOS)
            st.image(gray28, use_container_width=True, clamp=True)

        tensor = preprocess(image)
        pred_idx, letter, probs = predict(model, tensor)
        confidence = float(probs[pred_idx]) * 100

        st.divider()

        # Big prediction display
        res_left, res_right = st.columns([1, 2])
        with res_left:
            st.metric(label="Predicted letter", value=letter)
        with res_right:
            st.metric(label="Confidence", value=f"{confidence:.1f}%")

        # Top-5 predictions
        st.subheader("Top 5 predictions")
        top5_vals, top5_idxs = probs.topk(5)
        for rank, (idx, val) in enumerate(zip(top5_idxs, top5_vals), 1):
            ltr = label_to_letter(int(idx))
            pct = float(val) * 100
            st.progress(pct / 100, text=f"**{rank}. {ltr}** — {pct:.1f}%")


if __name__ == "__main__":
    main()
