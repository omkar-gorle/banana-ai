"""Production Streamlit UI for Banana AI.

Analysis is deliberately side-effect free.  A database write happens only
when the user presses ``Save prediction``.
"""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from banana_ai.config import settings
from banana_ai.ml.predict import load_model, predict_image


st.set_page_config(page_title="Banana AI", page_icon="🍌", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    """Load weights once per Streamlit process."""
    return load_model(model_path)


def _save_uploaded(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="reports") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def should_save_prediction(saved_path: str | None, current_path: str | None) -> bool:
    """Return false when the current Streamlit rerun already saved this upload."""
    return bool(current_path) and saved_path != current_path


st.title("🍌 Banana AI")
st.caption("Production ripeness classification · BananaCNN trained from scratch")

with st.sidebar:
    st.header("Model")
    st.code(settings.model_path)
    st.write(f"**Version:** `{settings.model_version}`")
    show_gradcam = st.checkbox("Show optional Grad-CAM", value=True)
    st.divider()
    st.header("Optional environment")
    temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 25.0, 0.5)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
    with st.expander("How it works"):
        st.write(
            "The image is resized to 224×224, normalized with ImageNet "
            "statistics, and classified by a four-class CNN. Probabilities "
            "are raw softmax scores and are not calibrated confidence."
        )
    with st.expander("About"):
        st.write("V2 was selected using validation macro-F1. V1 remains available "
                 "for historical evaluation only.")

uploaded = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    left, right = st.columns(2)
    with left:
        st.subheader("Image")
        st.image(uploaded, use_container_width=True)
    with right:
        st.subheader("Analysis")
        analyse = st.button("Analyse banana", type="primary", use_container_width=True)
        if analyse:
            if not Path(settings.model_path).exists():
                st.error(f"Production model not found: {settings.model_path}")
            else:
                path = _save_uploaded(uploaded)
                try:
                    with st.spinner("Classifying…"):
                        model, transform, classes, device = cached_model(settings.model_path)
                        result = predict_image(str(path), model, transform, classes, device)
                    st.session_state["prediction"] = result
                    st.session_state["prediction_path"] = str(path)
                    st.session_state["prediction_name"] = uploaded.name
                    st.session_state["saved_prediction_path"] = None
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Could not analyse image: {exc}")
                finally:
                    if st.session_state.get("prediction_path") != str(path):
                        path.unlink(missing_ok=True)

result = st.session_state.get("prediction")
if result:
    stage = result["stage"]
    st.divider()
    st.subheader(f"Prediction: {stage.upper()}")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Predicted stage", stage)
    metric2.metric("Raw softmax score", f"{result['confidence']:.1%}")
    metric3.metric("Model version", settings.model_version)
    st.caption("Scores are raw softmax outputs; they are not calibrated probabilities.")
    st.info(
        f"Prototype shelf-life estimate: **{result['estimated_days_left']}**. "
        "This is a heuristic based on stage, not a trained shelf-life model."
    )
    st.subheader("Class probabilities")
    for name, probability in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
        st.progress(probability, text=f"{name}: {probability:.1%}")

    if st.button("Save prediction", type="secondary"):
        if not should_save_prediction(
            st.session_state.get("saved_prediction_path"),
            st.session_state.get("prediction_path"),
        ):
            st.info("This prediction is already saved.")
        else:
            try:
                from banana_ai.db.crud import save_prediction
                from banana_ai.db.session import SessionLocal
                db = SessionLocal()
                try:
                    row = save_prediction(
                        db, st.session_state.get("prediction_name", "upload"),
                        stage, result["confidence"], result["estimated_days_left"],
                        settings.model_version, temperature, humidity,
                    )
                finally:
                    db.close()
                st.session_state["saved_prediction_path"] = st.session_state["prediction_path"]
                st.success(f"Prediction saved once (ID: {row.id}).")
            except Exception:
                st.warning("Prediction could not be saved. PostgreSQL remains optional for inference.")

    if show_gradcam:
        with st.expander("Optional Grad-CAM explanation"):
            try:
                from banana_ai.ml.explainability import generate_gradcam_overlay
                model, _, _, device = cached_model(settings.model_path)
                overlay, _, _, cam_confidence = generate_gradcam_overlay(
                    st.session_state["prediction_path"], model, device,
                )
                st.image(overlay, caption=f"Influential regions (score {cam_confidence:.1%})")
            except Exception as exc:
                st.info(f"Grad-CAM is unavailable for this image: {exc}")

st.divider()
st.subheader("Recent prediction history")
try:
    from banana_ai.db.crud import recent_predictions
    from banana_ai.db.session import SessionLocal
    db = SessionLocal()
    try:
        rows = recent_predictions(db)
        st.dataframe([
            {"ID": p.id, "Stage": p.predicted_stage, "Score": round(p.confidence, 4),
             "Model": p.model_version, "Created": str(p.created_at)[:19]}
            for p, _ in rows
        ], use_container_width=True)
    finally:
        db.close()
except Exception:
    st.info("Prediction history is unavailable while PostgreSQL is disconnected.")
