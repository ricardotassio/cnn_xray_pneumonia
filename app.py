import io
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import streamlit as st

# ---- NEW: remote loading ----
try:
    import requests
    import_requests = True
except ImportError:
    import_requests = False

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
luist = [
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/knn_(hog).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/svm_linear_svm_(hog).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/linear_svm_(cnnfeat).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/lr_(cnnfeat).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/lr_(hog).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/random_forest_(cnnfeat).pkl",
    "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/random_forest_(hog).pkl",

]
# -----------------------------
# Constants and utilities
# -----------------------------
APP_TITLE = "Chest X-Ray Pneumonia — Traditional Models (Pickle)"
IMG_SIZE = (224, 224)

# ---- NEW: your S3 base URL ----
S3_BASE_URL = "https://xrayprojectv1.s3.ca-central-1.amazonaws.com/"

def artifacts_dir() -> str:
    """
    Resolve ./artifacts relative to this file if available; otherwise cwd/artifacts.
    (Still used only to list model names locally if present.)
    """
    try:
        base = Path(__file__).parent
    except NameError:
        base = Path.cwd()
    return str(base / "artifacts")


def list_pickles_by_family(art_dir: str) -> Dict[str, List[str]]:
    """
    Build groups from S3 (via S3_BASE_URL ?list-type=2) or, if that fails,
    from the global array `luist`. Returns {'hog': [...], 'cnnfeat': [...]}.
    """
    groups = {"hog": [], "cnnfeat": []}
    names = []

    # --- Try listing directly from S3 (public ListBucket must be allowed) ---
    try:
        import requests
        import xml.etree.ElementTree as ET

        base = S3_BASE_URL.rstrip("/") + "/"
        url = base + "?list-type=2&max-keys=1000"
        r = requests.get(url, timeout=30)
        if r.ok:
            root = ET.fromstring(r.text)
            # Works with any namespace
            keys = [el.text for el in root.findall(".//{*}Contents/{*}Key")]
            names = [
                os.path.basename(k)
                for k in keys
                if isinstance(k, str) and k.lower().endswith(".pkl")
            ]
    except Exception:
        # ignore & fall back
        names = []

    # --- Fallback to `luist` (filenames or full URLs) ---
    if not names:
        try:
            raw = luist  # must exist elsewhere: e.g., ['knn_(hog).pkl', 'rf_(cnnfeat).pkl', ...]
        except NameError:
            raw = []
        seen = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            name = os.path.basename(item)
            if name.lower().endswith(".pkl") and name not in seen:
                seen.add(name)
                names.append(name)

    # Stable order
    names = sorted(names)

    # Group by family keyword
    for name in names:
        lower = name.lower()
        if "hog" in lower:
            groups["hog"].append(name)
        elif "cnnfeat" in lower or "cnn" in lower:
            groups["cnnfeat"].append(name)

    return groups

    """
    Build the model list from the global array `luist`.
    Each entry can be just a filename (e.g., 'knn_(hog).pkl') or a full URL.
    Groups by 'hog' and 'cnnfeat' (also accepts 'cnn' in the name).
    """
    groups = {"hog": [], "cnnfeat": []}

    # If `luist` isn't defined, return empty groups gracefully
    try:
        raw = luist  # must be defined elsewhere, e.g., luist = ['knn_(hog).pkl', ...]
    except NameError:
        return groups

    # Normalize entries to basenames and keep original order (dedup)
    seen = set()
    names: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = os.path.basename(item)
        if not name.lower().endswith(".pkl"):
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)

    # Group by family based on substring
    for name in names:
        lower = name.lower()
        if "hog" in lower:
            groups["hog"].append(name)
        elif "cnnfeat" in lower or "cnn" in lower:
            groups["cnnfeat"].append(name)

    return groups

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


# ---- NEW: download & cache from S3, then reuse the same loader ----
@st.cache_resource(show_spinner=False)
def load_pickle_model_from_url(url: str) -> BaseEstimator:
    if not import_requests:
        st.error("❌ Package 'requests' not found. Run: `pip install requests`")
        st.stop()

    # stream to temp file with a progress bar
    with requests.get(url, stream=True, timeout=180) as r:
        try:
            r.raise_for_status()
        except Exception as e:
            st.error(f"Failed to download model from S3.\nURL: {url}\nDetails: {e}")
            st.stop()

        total = int(r.headers.get("content-length", 0))
        done = 0
        prog = st.progress(0)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    tmp.write(chunk)
                    if total:
                        done += len(chunk)
                        prog.progress(min(int(done / total * 100), 100))
            tmp_path = tmp.name

        prog.empty()

    try:
        return load_pickle_model(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


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
st.sidebar.markdown("**Debug Info:**")
st.sidebar.text(f"Python: {sys.executable}")
st.sidebar.text(f"Version: {sys.version.split()[0]}")

# Check required imports
missing_imports = []
if not import_numpy:
    missing_imports.append("numpy")
if not import_pil:
    missing_imports.append("pillow")
if not import_sklearn:
    missing_imports.append("scikit-learn")
if not import_requests:
    missing_imports.append("requests")

if missing_imports:
    st.error(f"❌ Missing packages: {', '.join(missing_imports)}")
    st.info("Run: `pip install " + " ".join(missing_imports) + "`")
    st.stop()

# Model discovery (still by local names for the selector;
# the actual load will use S3_BASE_URL + selected filename)
art_dir = artifacts_dir()
groups = list_pickles_by_family(art_dir)

available_families = [opt for opt in ("hog", "cnnfeat") if groups[opt]]
if not available_families:
    st.warning("No .pkl models found locally in artifacts/. "
               "You can still run by adding at least placeholder files to list names, "
               "or adjust the code to provide a static list of remote filenames.")
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
    # ---- NEW: build S3 URL and load remotely ----
    pkl_url = S3_BASE_URL.rstrip("/") + "/" + model_name  # e.g., .../knn_(hog).pkl
    with st.spinner(f"Downloading model from S3…\n{pkl_url}"):
        est = load_pickle_model_from_url(pkl_url)

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
