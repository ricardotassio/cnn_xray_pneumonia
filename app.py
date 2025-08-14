import io
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import streamlit as st

# Lazy imports to avoid startup issues
try:
    import numpy as np
    import_numpy = True
except ImportError:
    import_numpy = False

try:
    from PIL import Image
    import_pil = True
except ImportError:
    import_pil = False

try:
    from sklearn.base import BaseEstimator
    import_sklearn = True
except ImportError:
    import_sklearn = False

# -----------------------------
# Constants and utilities
# -----------------------------
APP_TITLE = "Chest X-Ray Pneumonia — Traditional Models (Pickle)"
IMG_SIZE = (224, 224)


def artifacts_dir() -> str:
    """
    Resolve ./artifacts relative to this file if available; otherwise cwd/artifacts.
    """
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    return str(base / "artifacts")


def list_pickles_by_family(art_dir: str) -> Dict[str, List[str]]:
    """Return available .pkl files grouped by feature family (hog, cnnfeat)."""
    groups = {"hog": [], "cnnfeat": []}
    if not os.path.isdir(art_dir):
        return groups
    for name in sorted(os.listdir(art_dir)):
        if not name.lower().endswith(".pkl"):
            continue
        lower = name.lower()
        if "hog" in lower:
            groups["hog"].append(name)
        elif "cnnfeat" in lower or "cnn" in lower:
            groups["cnnfeat"].append(name)
    return groups


def _extract_estimator(obj):
    """Return the first object with a predict() method from possibly nested containers."""
    # Direct estimator
    if hasattr(obj, "predict"):
        return obj

    # Common wrapper attributes
    for attr in ("best_estimator_", "estimator_", "model", "clf", "pipeline", "pipe"):
        if hasattr(obj, attr):
            est = getattr(obj, attr)
            if hasattr(est, "predict"):
                return est

    # Dict
    if isinstance(obj, dict):
        for key in ("estimator", "model", "clf", "best_estimator_", "pipeline", "pipe"):
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]
        for v in obj.values():
            est = _extract_estimator(v)
            if est is not None:
                return est

    # List/Tuple
    if isinstance(obj, (list, tuple)):
        for v in obj:
            est = _extract_estimator(v)
            if est is not None:
                return est

    return None


@st.cache_resource(show_spinner=False)
def load_pickle_model(pkl_path: str) -> BaseEstimator:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    est = _extract_estimator(obj)
    if est is None:
        st.error(
            "Could not find a scikit-learn estimator with predict() in this pickle. "
            "Please ensure the pickle contains a fitted model."
        )
        st.stop()
    return est


def load_image(file: Union[bytes, io.BytesIO, Image.Image]) -> Image.Image:
    if isinstance(file, Image.Image):
        img = file
    else:
        img = Image.open(file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != IMG_SIZE:
        img = img.resize(IMG_SIZE)
    return img


def extract_hog_feature(img_rgb: Image.Image) -> np.ndarray:
    try:
        from skimage.color import rgb2gray  # lazy import
        from skimage.feature import hog      # lazy import
    except Exception as e:
        st.error(
            "scikit-image is required for HOG models. Install it with:\n"
            "python3 -m pip install scikit-image==0.25.2\n\n"
            f"Details: {e}"
        )
        st.stop()
    gray = rgb2gray(np.array(img_rgb))  # float64 in [0,1]
    feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


@st.cache_resource(show_spinner=False)
def get_efficientnet_b0_feature_extractor():
    """Return (EfficientNetB0 model, preprocess_input)."""
    try:
        import tensorflow as tf  # lazy import
    except Exception as e:
        st.error(
            "TensorFlow is required for CNN-feature models. Install it with:\n"
            "python3 -m pip install tensorflow==2.19.0\n\n"
            f"Details: {e}"
        )
        st.stop()

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3),
        pooling="avg",
    )
    # Access preprocess function through tf module to avoid direct import resolution issues
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    return base, preprocess_input


def extract_cnn_feature(img_rgb: Image.Image, extractor, preprocess_fn) -> np.ndarray:
    import tensorflow as tf
    arr = np.array(img_rgb).astype("float32")
    arr = preprocess_fn(arr)              # apply EfficientNet preprocessing
    arr = tf.expand_dims(arr, axis=0)     # shape (1, 224, 224, 3)
    feat = extractor.predict(arr, verbose=0)
    return np.asarray(feat).reshape(-1).astype(np.float32)


def predict_with_estimator(est: BaseEstimator, x: np.ndarray) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Predict label using sklearn estimator.
    Returns: (pred_label_int, prob_pos or None, raw_score or None)
    """
    x2d = x.reshape(1, -1)
    prob = None
    score = None
    # try probability first
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(x2d)
        pos_idx = 1
        if hasattr(est, "classes_"):
            classes = list(est.classes_)
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = int(np.argmax(classes))
        prob = float(proba[0, pos_idx])
    elif hasattr(est, "decision_function"):
        df = est.decision_function(x2d)
        score = float(df[0] if np.ndim(df) == 1 else df[0, 0])
    # Predict hard label
    pred = est.predict(x2d)
    pred_label = int(pred[0]) if isinstance(pred[0], (np.generic, int, float)) else int(pred[0])
    return pred_label, prob, score


def render_result(pred_label: int, prob: Optional[float], score: Optional[float]):
    label_text = "Pneumonia" if pred_label == 1 else "Normal"
    if pred_label == 1:
        st.error(f"Prediction: {label_text}")
    else:
        st.success(f"Prediction: {label_text}")
        if st.session_state.get("confetti_enabled", True):
            st.balloons()
    if prob is not None:
        st.write(f"Probability (positive=Pneumonia): {prob:.3f}")
        st.progress(int(min(max(prob, 0.0), 1.0) * 100))
    elif score is not None:
        st.write(f"Decision score: {score:.3f} (positive when > 0)")


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chest X-Ray Pneumonia", page_icon="🩺", layout="centered")
st.title(APP_TITLE)
st.caption("Upload a chest X-ray and classify it as Normal or Pneumonia using saved scikit-learn models.")

# Debug info
import sys
st.sidebar.markdown("**Debug Info:**")
st.sidebar.text(f"Python: {sys.executable}")
st.sidebar.text(f"Versão: {sys.version.split()[0]}")

# Check required imports
missing_imports = []
if not import_numpy:
    missing_imports.append("numpy")
if not import_pil:
    missing_imports.append("pillow")
if not import_sklearn:
    missing_imports.append("scikit-learn")

if missing_imports:
    st.error(f"❌ Pacotes necessários não encontrados: {', '.join(missing_imports)}")
    st.info("Execute: `pip install " + " ".join(missing_imports) + "`")
    st.stop()

art_dir = artifacts_dir()
groups = list_pickles_by_family(art_dir)

available_families = [opt for opt in ("hog", "cnnfeat") if groups[opt]]
if not available_families:
    st.warning("No .pkl models found in artifacts/. Please add the pickled models.")
    st.stop()

family = st.sidebar.radio(
    "Feature family",
    options=available_families,
    format_func=lambda k: "HOG-based" if k == "hog" else "CNN feature-based",
)

model_name = st.sidebar.selectbox(
    "Model",
    options=groups.get(family, []),
)

# UI option: enable/disable balloons on Normal predictions
st.session_state["confetti_enabled"] = st.sidebar.checkbox("🎈 Balloons on Normal", value=True)

uploaded = st.file_uploader("Choose an X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])

col_preview, col_action = st.columns([2, 1])
with col_preview:
    if uploaded is not None:
        img_preview = load_image(uploaded)
        st.image(img_preview, caption=f"Preview ({IMG_SIZE[0]}×{IMG_SIZE[1]})", use_container_width=True)

with col_action:
    do_predict = st.button("Predict", use_container_width=True, disabled=(uploaded is None or not model_name))

if do_predict and uploaded is not None and model_name:
    pkl_path = os.path.join(art_dir, model_name)
    with st.spinner("Loading model…"):
        est = load_pickle_model(pkl_path)

    img = load_image(uploaded)

    if family == "hog":
        with st.spinner("Extracting HOG features…"):
            feat = extract_hog_feature(img)
    else:
        with st.spinner("Extracting CNN features (EfficientNetB0)…"):
            try:
                extractor, preprocess_fn = get_efficientnet_b0_feature_extractor()
            except Exception as e:
                st.error(
                    "Failed to initialize EfficientNetB0 feature extractor. "
                    "Ensure TensorFlow is installed and internet access is available for first-time ImageNet weights download.\n\n"
                    f"Details: {e}"
                )
                st.stop()
            feat = extract_cnn_feature(img, extractor, preprocess_fn)

    with st.spinner("Predicting…"):
        pred_label, prob, score = predict_with_estimator(est, feat)
    render_result(pred_label, prob, score)

st.markdown("""
**Notes**
- HOG models were trained on 224×224 grayscale images with `orientations=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`, `block_norm='L2-Hys'`.
- CNN-feature models use EfficientNetB0 (ImageNet weights, global average pooling) as an embedding extractor with proper `preprocess_input`.
- Label convention: `0=Normal`, `1=Pneumonia`.
""")
