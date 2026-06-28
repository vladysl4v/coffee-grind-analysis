import streamlit as st
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms

from src.models.convnext import get_convnext_small
from src.data_loader import DEFAULT_EVAL_TRANSFORM, _NORMALIZE


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL1_PATH = "data/runs/convnext_small/run_001/models/best_model.pt"
MODEL2_PATH = "data/runs/convnext_small/run_002/models/best_model.pt"

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
def load_model(model_path):
    model = get_convnext_small(freeze_backbone=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


model1 = load_model(MODEL1_PATH)
model2 = load_model(MODEL2_PATH)

transform1 = DEFAULT_EVAL_TRANSFORM

transform2 = transforms.Compose([
    transforms.CenterCrop(420),
    transforms.ToTensor(),
    _NORMALIZE,
])


def init_state():
    if "pred1" not in st.session_state:
        st.session_state.pred1 = None

    if "pred2" not in st.session_state:
        st.session_state.pred2 = None

    if "actual_input" not in st.session_state:
        st.session_state.actual_input = None

    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None


def reset_state():
    st.session_state.pred1 = None
    st.session_state.pred2 = None
    st.session_state.actual_input = None


init_state()

@torch.no_grad()
def predict(model, transform, image: Image.Image):
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, width=320)

    with col2:

        actual = get_actual_value(filename)

        st.write(f"**File:** {filename}")

        if st.button("Predict Fineness"):
            st.session_state.pred1 = predict(model1, transform1, image)
            st.session_state.pred2 = predict(model2, transform2, image)

        if st.session_state.pred1 is not None:

            st.subheader("Results")

            model_col1, model_col2 = st.columns(2)

            with model_col1:
                st.markdown("### Model 1")
                st.caption("Trained on usual dataset")
                st.metric(
                    "Predicted Fineness",
                    f"{st.session_state.pred1:.2f}"
                )

            with model_col2:
                st.markdown("### Model 2")
                st.caption("Trained with synthetic data")
                st.metric(
                    "Predicted Fineness",
                    f"{st.session_state.pred2:.2f}"
                )

            if actual is not None:

                st.subheader("Comparison")

                st.metric("Actual Fineness", f"{actual:.2f}")

                err_col1, err_col2 = st.columns(2)

                err1 = abs(st.session_state.pred1 - actual)
                err2 = abs(st.session_state.pred2 - actual)

                with err_col1:
                    st.metric("Model 1 Error", f"{err1:.2f}")

                with err_col2:
                    st.metric("Model 2 Error", f"{err2:.2f}")

                if err1 < err2:
                    st.success("Model 1 is closer to the actual value.")
                elif err2 < err1:
                    st.success("Model 2 is closer to the actual value.")
                else:
                    st.info("Both models have the same error.")

            else:
                st.warning("No label found in dataset.")

                st.session_state.actual_input = st.number_input(
                    "Enter actual fineness",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key=f"manual_{filename}"
                )

                if st.session_state.actual_input is not None:

                    actual_manual = st.session_state.actual_input

                    st.subheader("Comparison")

                    man_col1, man_col2 = st.columns(2)

                    with man_col1:
                        st.metric(
                            "Model 1 Error",
                            f"{abs(st.session_state.pred1 - actual_manual):.2f}"
                        )

                    with man_col2:
                        st.metric(
                            "Model 2 Error",
                            f"{abs(st.session_state.pred2 - actual_manual):.2f}"
                        )

            st.subheader("Predicted Fineness")

            prog_col1, prog_col2 = st.columns(2)

            with prog_col1:
                st.write("Model 1")
                st.progress(int(max(0, min(st.session_state.pred1, 100))))

                if st.session_state.pred1 < 30:
                    st.info("Coarse Grind")
                elif st.session_state.pred1 < 70:
                    st.warning("Medium Grind")
                else:
                    st.success("Fine Grind")

            with prog_col2:
                st.write("Model 2")
                st.progress(int(max(0, min(st.session_state.pred2, 100))))

                if st.session_state.pred2 < 30:
                    st.info("Coarse Grind")
                elif st.session_state.pred2 < 70:
                    st.warning("Medium Grind")
                else:
                    st.success("Fine Grind")

else:
    st.info("Upload an image to start prediction.")