import streamlit as st
import torch
import pandas as pd
from PIL import Image
from pathlib import Path

from src.models.convnext import get_convnext_small
from src.data_loader import DEFAULT_EVAL_TRANSFORM


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "data/runs/convnext_small/run_001/models/best_model.pt"

LABEL_FILES = [
    "data/labels/train.csv",
    "data/labels/val.csv",
    "data/labels/test.csv",
]

@st.cache_data
def load_all_labels():
    dfs = []
    for path in LABEL_FILES:
        df = pd.read_csv(path, sep=";")

        df["Fineness"] = (
            df["Fineness"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

labels_df = load_all_labels()


def get_actual_value(filename: str):
    filename = Path(filename).name
    match = labels_df[labels_df["Sample"] == filename]
    return float(match.iloc[0]["Fineness"]) if not match.empty else None


@st.cache_resource
def load_model():
    model = get_convnext_small(freeze_backbone=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = DEFAULT_EVAL_TRANSFORM

def init_state():
    if "pred" not in st.session_state:
        st.session_state.pred = None

    if "actual_input" not in st.session_state:
        st.session_state.actual_input = None

    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None


def reset_state():
    st.session_state.pred = None
    st.session_state.actual_input = None


init_state()

@torch.no_grad()
def predict(image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)
    pred = model(x).squeeze().cpu().item()
    return float(pred) * 100


st.set_page_config(
    page_title="Coffee Grind Predictor",
    page_icon="☕",
    layout="wide",
)

st.title("☕ Coffee Grind Fineness Predictor")

uploaded_file = st.file_uploader(
    "Upload coffee image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file:

    filename = uploaded_file.name

    if st.session_state.last_filename != filename:
        reset_state()
        st.session_state.last_filename = filename

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, width=320)

    with col2:

        actual = get_actual_value(filename)

        st.write(f"**File:** {filename}")

        if st.button("Predict Fineness"):
            st.session_state.pred = predict(image)

        if st.session_state.pred is not None:

            st.subheader("Results")

            st.metric("Predicted Fineness", f"{st.session_state.pred:.2f}")

            if actual is not None:
                st.metric("Actual Fineness", f"{actual:.2f}")
                st.metric("Absolute Error", f"{abs(st.session_state.pred - actual):.2f}")

            else:
                st.warning("No label found in dataset.")

                st.session_state.actual_input = st.number_input(
                    "Enter actual fineness",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key=f"manual_{filename}"
                )

                if st.button("Compare"):
                    pass  # value already stored


            if st.session_state.actual_input is not None:

                st.success("Comparison")

                st.metric("Manual Actual", f"{st.session_state.actual_input:.2f}")
                st.metric(
                    "Error",
                    f"{abs(st.session_state.pred - st.session_state.actual_input):.2f}"
                )

            progress = int(max(0, min(st.session_state.pred, 100)))
            st.progress(progress)

            if st.session_state.pred < 30:
                st.info("Coarse Grind")
            elif st.session_state.pred < 70:
                st.warning("☕ Medium Grind")
            else:
                st.success("Fine Grind")

else:
    st.info("Upload an image to start prediction.")