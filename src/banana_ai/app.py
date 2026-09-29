"""Banana AI -- Banana Quality & Shelf-Life Intelligence System.

Streamlit UI with:
  - Camera capture + file upload (both always available)
  - Prediction result card with confidence and class probabilities
  - Environment-aware shelf-life estimation (prototype heuristic)
  - What-if environment simulator
  - Grad-CAM explanation
  - Human feedback
  - Prediction history dashboard
  - Analytics dashboard
  - Batch analysis
  - Food-waste insights
  - Eat-first priority
  - Scan report download (HTML + CSV)
  - About the AI section

Analysis is deliberately side-effect free.
A database write happens ONLY when the user presses "Save prediction".
Batch results are saved only when the user presses "Save Batch Results".
"""
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Streamlit Cloud executes this file directly, so the src-layout package
# directory is not guaranteed to be on Python's import path.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from PIL import Image

from banana_ai.config import settings
from banana_ai.ml.predict import load_model, predict_image
from banana_ai.services.shelf_life import (
    STAGE_RANGES,
    STORAGE_LABELS,
    estimate_shelf_life,
)
from banana_ai.services.image_validation import (
    ImageValidationError,
    validate_image_bytes,
    safe_temp_filename,
)
from banana_ai.services.banana_validation import (
    validate_banana_image,
    ValidationResult,
    ValidationState,
)
from banana_ai.services.report import (
    generate_scan_report_html,
    generate_scan_report_csv,
    generate_batch_csv,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config -- MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Banana AI -- Quality & Shelf-Life Intelligence",
    page_icon="U+1F34C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Mobile-friendly CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
.hero { background: linear-gradient(135deg, #17211b 0%, #243b2a 58%, #3d542e 100%); border: 1px solid rgba(255, 214, 74, .22); border-radius: 24px; padding: 34px 38px; margin-bottom: 22px; color: #f8faf7; box-shadow: 0 18px 40px rgba(20, 35, 25, .18); }
.hero-kicker { color: #f6d34a; font-size: .78rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { color: #fffdf2; font-size: clamp(2rem, 5vw, 3.3rem); line-height: 1.05; margin: 10px 0; }
.hero p { color: #dce8d8; font-size: 1.05rem; max-width: 640px; margin-bottom: 18px; }
.online-badge { display: inline-block; color: #b8f2c2; background: rgba(71, 184, 99, .15); border: 1px solid rgba(130, 230, 145, .3); border-radius: 999px; padding: 6px 12px; font-size: .78rem; font-weight: 700; }
.step-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0 24px; }
.step { flex: 1 1 150px; background: #f7f8f4; border: 1px solid #e5e9df; border-radius: 14px; padding: 12px 14px; color: #526052; font-size: .86rem; }
.step strong { display: block; color: #233326; font-size: .92rem; margin-bottom: 3px; }
.step-active { background: #fff9dd; border-color: #f0d45c; }
.input-card { background: #ffffff; border: 1px solid #e5e9df; border-radius: 20px; padding: 24px; box-shadow: 0 10px 28px rgba(31, 45, 35, .06); }
.input-card h2 { color: #243326; margin-top: 0; }
.eyebrow { color: #8a7620; font-size: .74rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.empty-state { text-align: center; background: #fbfcf9; border: 1px dashed #cfd9cc; border-radius: 18px; padding: 28px 18px; color: #667466; }
.empty-icon { font-size: 2.5rem; }
.section-title { color: #243326; margin: 28px 0 10px; }
.tech-note { color: #687568; font-size: .83rem; }
.stButton > button, .stDownloadButton > button { border-radius: 11px; font-weight: 650; min-height: 2.6rem; }
.result-panel { background: #fff; border: 1px solid #e5e9df; border-radius: 20px; padding: 22px; box-shadow: 0 10px 28px rgba(31, 45, 35, .06); height: 100%; }
.result-label { color: #728071; font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.result-stage { font-size: clamp(2.4rem, 6vw, 4.5rem); font-weight: 800; line-height: 1; margin: 12px 0 8px; }
.result-confidence { color: #324335; font-size: 1.05rem; }
.confidence-track { background: #edf1e9; border-radius: 999px; height: 10px; margin: 14px 0 6px; overflow: hidden; }
.confidence-fill { background: linear-gradient(90deg, #d4ae27, #f5d85d); border-radius: inherit; height: 100%; }
.metric-card { background: #f8faf6; border: 1px solid #e7ece3; border-radius: 15px; padding: 15px; }
.metric-card .metric-value { color: #26382a; font-size: 1.35rem; font-weight: 750; }
.metric-card .metric-label { color: #718071; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
.prob-row { margin: 10px 0; }
.prob-head { display: flex; justify-content: space-between; color: #435246; font-size: .86rem; }
.prob-track { background: #edf1e9; border-radius: 999px; height: 8px; margin-top: 5px; overflow: hidden; }
.prob-fill { background: #d8b629; border-radius: inherit; height: 100%; }
@media (max-width: 640px) {
    .hero { padding: 26px 22px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .result-stage { font-size: 3rem; }
}
.prediction-card {
    background: linear-gradient(135deg, #121829 0%, #1e293b 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    border-radius: 18px;
    padding: 28px;
    color: white;
    text-align: center;
    margin: 16px 0;
}
.stage-ripe    { color: #4ade80; font-size: 2.2rem; font-weight: 700; text-shadow: 0 0 20px rgba(74, 222, 128, 0.3); }
.stage-overripe { color: #fbbf24; font-size: 2.2rem; font-weight: 700; text-shadow: 0 0 20px rgba(251, 191, 36, 0.3); }
.stage-rotten  { color: #f87171; font-size: 2.2rem; font-weight: 700; text-shadow: 0 0 20px rgba(248, 113, 113, 0.3); }
.stage-unripe  { color: #60a5fa; font-size: 2.2rem; font-weight: 700; text-shadow: 0 0 20px rgba(96, 165, 250, 0.3); }
.gate-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 6px;
}
.gate-badge-pass { background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
.gate-badge-fail { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model caching -- load once per process (FEATURE 24)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_model(model_path: str):
    """Load model weights once per Streamlit process (CPU-friendly)."""
    return load_model(model_path)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_ANALYZE = "Analyze"
PAGE_BATCH = "Batch"
PAGE_HISTORY = "History"
PAGE_ANALYTICS = "Analytics"
PAGE_WASTE = "Waste Insights"
PAGE_ABOUT = "About AI"

_ASSESSMENT: dict[str, str] = {
    "ripe": (
        "Likely ready to eat. Best consumed within the estimated window."
    ),
    "overripe": (
        "Very soft and sweet -- still edible for most people but deteriorating. "
        "Consume or use in cooking soon."
    ),
    "rotten": (
        "Model classified this banana as rotten. Inspect carefully before "
        "consumption. Do not treat this model as a food-safety guarantee."
    ),
    "unripe": (
        "Still developing. Allow to ripen at room temperature. "
        "Check back in a few days."
    ),
}

_PRIORITY_LABELS: dict[str, str] = {
    "overripe": "High priority -- consume or use very soon.",
    "ripe": "Medium priority -- consume within estimated window.",
    "unripe": "Low priority -- allow to ripen first.",
    "rotten": "Rotten -- inspect and discard per normal food-safety judgment.",
}

_STAGE_EMOJI: dict[str, str] = {
    "ripe": "[RIPE]",
    "overripe": "[OVERRIPE]",
    "rotten": "[ROTTEN]",
    "unripe": "[UNRIPE]",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_and_save_temp(
    data: bytes, filename: str
) -> tuple[Optional[Path], Optional[ValidationResult]]:
    """Validate image bytes, run banana content check, save to temp file.

    Returns
    -------
    (path, banana_result)
        path is None when any validation step fails.
        banana_result is always returned so the caller can inspect state/confidence.
    """
    # Step 1 – file / format validation
    try:
        img = validate_image_bytes(data, filename=filename)
    except ImageValidationError as exc:
        st.error(f"Image rejected: {exc}")
        return None, None

    # Step 2 – banana content validation
    banana_result = validate_banana_image(img)

    if banana_result.state == ValidationState.NOT_BANANA:
        st.markdown(
            """
<div style="background:linear-gradient(135deg, #2b1212 0%, #1a0a0a 100%);border:1px solid #ef5350;border-radius:14px;padding:22px 24px;margin:16px 0;box-shadow:0 8px 24px rgba(239,83,80,0.18)">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
    <span style="font-size:1.8rem">&#9940;</span>
    <h3 style="color:#ef5350;margin:0;font-size:1.35rem;font-weight:700">Image Blocked by Banana Gate</h3>
  </div>
  <p style="color:#ffcdd2;font-size:1.0rem;margin:0 0 10px;line-height:1.5">
    The dedicated banana gate classifier verified that this image does <strong>not</strong> contain a banana.
  </p>
  <div style="background:rgba(255,255,255,0.06);border-left:3px solid #ef5350;border-radius:6px;padding:10px 14px;margin:10px 0;color:#ffcdd2;font-size:0.88rem">
    &#128737;&#65039; <strong>Pipeline Protection:</strong> Ripeness classification, Grad-CAM visualization, and shelf-life prediction are blocked to protect downstream accuracy.
  </div>
  <p style="color:#ffcdd2;margin:10px 0 0;font-size:0.9rem">
    Please capture or upload a clear photo of a real banana (unripe, ripe, overripe, or rotten).
  </p>
  <div style="margin-top:14px;display:flex;align-items:center;gap:8px">
    <span class="gate-badge gate-badge-fail">Gate Confidence: {conf:.1%}</span>
    <span style="color:#9e9e9e;font-size:0.8rem">(Threshold: {threshold:.1%})</span>
  </div>
</div>""".format(
    conf=banana_result.confidence,
    threshold=settings.banana_gate_threshold,
),
            unsafe_allow_html=True,
        )
        return None, banana_result

    if banana_result.state == ValidationState.UNCERTAIN:
        st.markdown(
            """
<div style="background:linear-gradient(135deg, #2b1d0a 0%, #1a1205 100%);border:1px solid #ffa726;border-radius:14px;padding:22px 24px;margin:16px 0;box-shadow:0 8px 24px rgba(255,167,38,0.18)">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
    <span style="font-size:1.8rem">&#9888;&#65039;</span>
    <h3 style="color:#ffa726;margin:0;font-size:1.35rem;font-weight:700">Banana Gate: Ambiguous Content</h3>
  </div>
  <p style="color:#ffe0b2;font-size:1.0rem;margin:0 0 10px;line-height:1.5">
    The image is unclear or the banana is not sufficiently visible for confident gating.
  </p>
  <ul style="color:#ffe0b2;margin:6px 0 0 20px;font-size:0.9rem;line-height:1.6">
    <li>Move the camera closer to the banana</li>
    <li>Improve ambient lighting and avoid dark shadows</li>
    <li>Ensure the banana is centered and not obstructed</li>
  </ul>
  <div style="margin-top:14px;display:flex;align-items:center;gap:8px">
    <span class="gate-badge gate-badge-fail" style="background:rgba(255,167,38,0.15);color:#ffa726;border-color:rgba(255,167,38,0.3)">Gate Confidence: {conf:.1%}</span>
    <span style="color:#9e9e9e;font-size:0.8rem">(Threshold: {threshold:.1%})</span>
  </div>
</div>""".format(
    conf=banana_result.confidence,
    threshold=settings.banana_gate_threshold,
),
            unsafe_allow_html=True,
        )
        return None, banana_result

    # Banana confirmed — save to temp file
    safe_name = safe_temp_filename(filename)
    tmp_dir = Path("reports")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / safe_name

    with tmp_path.open("wb") as f:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        f.write(img_bytes.getvalue())

    return tmp_path, banana_result


def _run_prediction(image_path: Path):
    """Run the V2 classifier and return result dict."""
    if not Path(settings.model_path).exists():
        st.error(f"Production model not found: {settings.model_path}")
        return None
    with st.spinner("Classifying banana..."):
        model, transform, classes, device = cached_model(settings.model_path)
        return predict_image(str(image_path), model, transform, classes, device)


def should_save_prediction(saved_path: Optional[str], current_path: Optional[str]) -> bool:
    """Return False when this upload was already saved this session."""
    return bool(current_path) and saved_path != current_path


def _clear_prediction_state() -> None:
    """Clear result state when the user starts a different analysis."""
    for key in (
        "prediction",
        "prediction_path",
        "prediction_name",
        "saved_prediction_path",
        "banana_detection_confidence",
        "feedback_given",
        "show_correction",
    ):
        st.session_state[key] = None


def _get_db():
    """Open a DB session. Returns None if DB unavailable."""
    try:
        from banana_ai.db.session import SessionLocal
        return SessionLocal()
    except Exception:
        return None


def _save_feedback_to_db(feedback: str, corrected_stage: Optional[str]) -> None:
    """Save feedback to DB if a prediction was saved this session."""
    pred_id = st.session_state.get("last_saved_prediction_id")
    if not pred_id:
        st.warning("Save the prediction first before giving feedback.")
        return
    db = _get_db()
    if db is None:
        st.warning("PostgreSQL unavailable -- feedback not stored.")
        return
    try:
        from banana_ai.db.crud import save_feedback as db_fb
        db_fb(db=db, prediction_id=pred_id, feedback=feedback,
              corrected_stage=corrected_stage)
        st.session_state["feedback_given"] = True
        st.session_state["show_correction"] = False
        st.success("Feedback recorded. Thank you! (This will NOT retrain the model.)")
    except Exception as exc:
        st.warning(f"Could not save feedback: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🍌 Banana AI")
    st.caption("Smart vision for banana quality")
    st.divider()

    st.markdown("### Environment (optional)")
    temperature = st.number_input(
        "Temperature (deg C)", -20.0, 60.0, 25.0, 0.5, key="temp_input"
    )
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, 60.0, 1.0, key="hum_input"
    )
    storage_condition = st.selectbox(
        "Storage condition",
        options=STORAGE_LABELS,
        format_func=lambda x: {
            "room": "Room / Ambient",
            "cool": "Cool Storage",
            "refrigerator": "Refrigerator",
            "other": "Other",
        }.get(x, x),
        key="storage_input",
    )

    st.divider()
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

    st.divider()
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        options=[PAGE_ANALYZE, PAGE_BATCH, PAGE_HISTORY,
                 PAGE_ANALYTICS, PAGE_WASTE, PAGE_ABOUT],
        key="nav_page",
        label_visibility="collapsed",
    )

    with st.expander("How it works"):
        st.write(
            "Images are resized to 224x224, normalized with ImageNet statistics, "
            "and classified by banana-cnn-v2 (custom CNN, ~422,788 params). "
            "Probabilities are raw softmax scores -- not calibrated confidence."
        )
    with st.expander("Technical details"):
        st.code(settings.model_path, language=None)
        st.write(f"**Version:** `{settings.model_version}`")
        st.caption("Production checkpoints are frozen and loaded through the cached inference path.")


# ===========================================================================
# PAGE: Analyze (FEATURES 1-12, 15-18)
# ===========================================================================

def page_analyze():
    st.markdown(
        """
        <section class="hero">
          <div class="hero-kicker">🍌 Banana AI · Computer Vision</div>
          <h1>Smart vision for banana quality.</h1>
          <p>Analyze ripeness, understand the result, and estimate a quality window with an explainable AI workflow.</p>
          <span class="online-badge">● AI SYSTEM ONLINE</span>
        </section>
        <div class="step-strip">
          <div class="step step-active"><strong>1 · Capture</strong>Upload or take a photo.</div>
          <div class="step"><strong>2 · Validate</strong>Banana gate checks the image.</div>
          <div class="step"><strong>3 · Analyze</strong>Ripeness model classifies it.</div>
          <div class="step"><strong>4 · Insights</strong>Explanation and shelf-life.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # FEATURE 1 -- Dual input
    tab_camera, tab_upload = st.tabs(["Take Photo", "Upload Photo"])

    active_image_bytes: Optional[bytes] = None
    active_filename: str = ""
    active_input_method: str = "upload"

    with tab_camera:
        st.markdown(
            '<div class="input-card"><div class="eyebrow">Step 1 · Capture image</div><h2>Take a photo</h2><p class="tech-note">Place a banana clearly inside the frame, in good light.</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Camera input works in modern browsers that support camera access. "
            "If camera is unavailable or permission is denied, use Upload Photo."
        )
        camera_image = st.camera_input(
            "Capture banana photo",
            help="Point your camera at the banana and press the capture button.",
            key="camera_input",
        )
        if camera_image is not None:
            active_image_bytes = camera_image.getvalue()
            active_filename = "camera_capture.jpg"
            active_input_method = "camera"
            st.success("Photo captured! Press Analyze Banana below.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_upload:
        st.markdown(
            '<div class="input-card"><div class="eyebrow">Step 1 · Capture image</div><h2>Upload a photo</h2><p class="tech-note">JPG, PNG, or WEBP · Keep the banana centered and visible.</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose a banana image",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, JPEG, PNG, or WEBP. Maximum 20 MB.",
            key="upload_input",
        )
        if uploaded_file is not None:
            active_image_bytes = uploaded_file.getvalue()
            active_filename = uploaded_file.name
            active_input_method = "upload"
            st.success(f"File loaded: {uploaded_file.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Image preview + analyze button
    if active_image_bytes:
        if active_filename != st.session_state.get("prediction_name"):
            _clear_prediction_state()
        st.divider()
        col_preview, col_analyze = st.columns([1, 1])

        with col_preview:
            st.subheader("Image Preview")
            try:
                preview_img = Image.open(io.BytesIO(active_image_bytes)).convert("RGB")
                st.image(preview_img, use_container_width=True)
            except Exception:
                st.warning("Could not display image preview.")

        with col_analyze:
            st.subheader("Analysis")
            analyze_btn = st.button(
                "Analyze Banana",
                type="primary",
                use_container_width=True,
                key="analyze_btn",
            )

            if analyze_btn:
                # Isolate state: clear previous results immediately so rejected images never show stale data
                _clear_prediction_state()

                image_path, banana_result = _validate_and_save_temp(
                    active_image_bytes, active_filename
                )
                if image_path and banana_result:
                    try:
                        result = _run_prediction(image_path)
                        if result:
                            st.session_state["prediction"] = result
                            st.session_state["prediction_path"] = str(image_path)
                            st.session_state["prediction_name"] = active_filename
                            st.session_state["prediction_input_method"] = active_input_method
                            st.session_state["saved_prediction_path"] = None
                            st.session_state["feedback_given"] = False
                            st.session_state["banana_detection_confidence"] = banana_result.confidence
                    except (OSError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not analyse image: {exc}")
                        logger.exception("Prediction error")
    elif not st.session_state.get("prediction"):
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">🍌</div>
              <strong>Your banana analysis will appear here.</strong>
              <div>Upload or capture an image to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Results
    result = st.session_state.get("prediction")
    if result:
        stage = result["stage"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        input_method = st.session_state.get("prediction_input_method", "upload")

        st.markdown('<h2 class="section-title">Analysis complete</h2>', unsafe_allow_html=True)

        detection_conf = st.session_state.get("banana_detection_confidence", 0.0)
        preview_path = st.session_state.get("prediction_path")
        result_col, image_col = st.columns([1.15, 0.85], gap="large")
        with result_col:
            st.markdown(f"""
            <div class="result-panel">
              <div class="result-label">🍌 AI analysis complete</div>
              <div class="result-stage stage-{stage}">{stage.upper()}</div>
              <div class="result-confidence"><strong>{confidence:.1%}</strong> AI confidence</div>
              <div class="confidence-track"><div class="confidence-fill" style="width:{confidence:.1%}"></div></div>
              <div class="tech-note">Raw softmax score, not a calibrated probability.</div>
              <div class="metric-card" style="margin-top:18px">
                <div class="metric-label">Banana gate</div>
                <div class="metric-value">{detection_conf:.1%} detected</div>
                <div class="tech-note">Validated before ripeness analysis.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with image_col:
            st.markdown('<div class="result-panel"><div class="result-label">Your image</div>', unsafe_allow_html=True)
            if preview_path and Path(preview_path).exists():
                st.image(Image.open(preview_path).convert("RGB"), use_container_width=True)
            st.caption(f"{input_method.capitalize()} · ready for review")
            st.markdown("</div>", unsafe_allow_html=True)

        # FEATURE 3 -- Quality assessment
        assessment = _ASSESSMENT.get(stage, "Assessment unavailable.")
        st.info(f"**Assessment:** {assessment}")

        # FEATURE 15 -- Eat-first priority
        priority_label = _PRIORITY_LABELS.get(stage, "")
        if priority_label:
            st.markdown(f"**Eat-First Priority:** {priority_label}")

        st.markdown('<h3 class="section-title">Ripeness distribution</h3>', unsafe_allow_html=True)
        probability_html = []
        for name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            probability_html.append(
                f'<div class="prob-row"><div class="prob-head"><span>{name.capitalize()}</span><strong>{prob:.1%}</strong></div>'
                f'<div class="prob-track"><div class="prob-fill" style="width:{prob:.1%}"></div></div></div>'
            )
        st.markdown(
            '<div class="result-panel">' + "".join(probability_html)
            + '<div class="tech-note">Raw softmax scores. All four scores sum to approximately 100%.</div></div>',
            unsafe_allow_html=True,
        )

        # FEATURE 4 & 5 -- Environment-aware shelf-life
        st.markdown('<h3 class="section-title">Quality insights</h3>', unsafe_allow_html=True)

        shelf_life = estimate_shelf_life(
            predicted_stage=stage,
            temperature_c=temperature,
            humidity_pct=humidity,
            storage_condition=storage_condition,
        )

        if stage == "rotten":
            st.error("Stage is rotten -- no quality window estimated.")
        else:
            col_sl1, col_sl2 = st.columns(2)
            with col_sl1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">⏳ Prototype shelf life</div><div class="metric-value">~{shelf_life.display()}</div><div class="tech-note">Heuristic estimate</div></div>',
                    unsafe_allow_html=True,
                )
            with col_sl2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">🌡 Environment</div><div class="metric-value">{temperature:.0f}°C · {humidity:.0f}%</div><div class="tech-note">{storage_condition.capitalize()} storage</div></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("How was this estimated?", expanded=False):
                st.write(shelf_life.explanation)
                st.caption(
                    "Prototype Heuristic -- coefficients are prototype assumptions, "
                    "NOT experimentally validated scientific constants."
                )

        st.warning(shelf_life.warning)

        with st.expander("Technical details"):
            st.write({
                "Model": settings.model_version,
                "Banana gate": "MobileNetV3-Small",
                "Input": "224 x 224 RGB",
                "Processing": "Computer vision",
                "Input method": input_method.capitalize(),
                "Banana gate confidence": f"{detection_conf:.1%}",
            })

        # FEATURE 6 -- What-if simulator
        with st.expander("WHAT IF? -- Environment Simulator"):
            st.caption("Prototype simulation -- uses the same heuristic formula.")
            wif_col1, wif_col2 = st.columns(2)
            with wif_col1:
                wif_temp = st.slider(
                    "Hypothetical temperature (deg C)", -20.0, 60.0,
                    float(temperature), 1.0, key="wif_temp"
                )
            with wif_col2:
                wif_hum = st.slider(
                    "Hypothetical humidity (%)", 0.0, 100.0,
                    float(humidity), 1.0, key="wif_hum"
                )
            wif_storage = st.selectbox(
                "Hypothetical storage",
                options=STORAGE_LABELS,
                format_func=lambda x: {
                    "room": "Room",
                    "cool": "Cool Storage",
                    "refrigerator": "Refrigerator",
                    "other": "Other",
                }.get(x, x),
                key="wif_storage",
            )
            wif_sl = estimate_shelf_life(
                predicted_stage=stage,
                temperature_c=wif_temp,
                humidity_pct=wif_hum,
                storage_condition=wif_storage,
            )
            wif_col3, wif_col4 = st.columns(2)
            with wif_col3:
                st.metric("Current estimate", f"~{shelf_life.display()}")
            with wif_col4:
                st.metric("Hypothetical estimate", f"~{wif_sl.display()}")
            st.caption(wif_sl.explanation)

        # FEATURE 8 -- Grad-CAM
        st.divider()
        if show_gradcam:
            with st.expander("Why did the model predict this? (Grad-CAM)"):
                st.markdown(
                    "**The highlighted regions show image areas that contributed most "
                    "to the model's prediction.** Grad-CAM is an explanation aid -- "
                    "it does NOT prove the banana is rotten/ripe/etc."
                )
                try:
                    from banana_ai.ml.explainability import generate_gradcam_overlay
                    model_obj, _, _, device = cached_model(settings.model_path)
                    overlay, _, _, cam_confidence = generate_gradcam_overlay(
                        st.session_state["prediction_path"], model_obj, device,
                    )
                    gc_col1, gc_col2 = st.columns(2)
                    with gc_col1:
                        pred_path = st.session_state.get("prediction_path")
                        if pred_path and Path(pred_path).exists():
                            st.image(
                                Image.open(pred_path).convert("RGB"),
                                caption="Original image",
                                use_container_width=True,
                            )
                    with gc_col2:
                        st.image(
                            overlay,
                            caption=f"Grad-CAM overlay (score {cam_confidence:.1%})",
                            use_container_width=True,
                        )
                    st.caption(
                        "Highlighted regions are influential -- not proof of the prediction. "
                        "Grad-CAM is a visualization tool, not ground-truth evidence."
                    )
                except Exception as exc:
                    st.info(f"Grad-CAM unavailable for this image: {exc}")
                    logger.info("Grad-CAM unavailable: %s", exc)

        # FEATURE 16 -- Report download
        st.divider()
        st.subheader("Download Scan Report")
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            html_report = generate_scan_report_html(
                prediction=result,
                temperature_c=temperature,
                humidity_pct=humidity,
                storage_condition=storage_condition,
                shelf_life=shelf_life,
                input_method=st.session_state.get("prediction_input_method", "upload"),
                model_version=settings.model_version,
            )
            st.download_button(
                "Download HTML Report",
                data=html_report,
                file_name="banana_ai_scan_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with rep_col2:
            csv_report = generate_scan_report_csv(
                prediction=result,
                temperature_c=temperature,
                humidity_pct=humidity,
                storage_condition=storage_condition,
                shelf_life=shelf_life,
                input_method=st.session_state.get("prediction_input_method", "upload"),
                model_version=settings.model_version,
            )
            st.download_button(
                "Download CSV Report",
                data=csv_report,
                file_name="banana_ai_scan_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # FEATURE 19 -- Save prediction (explicit only)
        st.divider()
        st.subheader("Save Prediction")
        if st.button("Save Prediction", type="secondary", key="save_pred_btn"):
            if not should_save_prediction(
                st.session_state.get("saved_prediction_path"),
                st.session_state.get("prediction_path"),
            ):
                st.info("This prediction is already saved this session.")
            else:
                db = _get_db()
                if db is None:
                    st.warning("PostgreSQL unavailable -- prediction not saved.")
                else:
                    try:
                        from banana_ai.db.crud import save_prediction as db_save
                        row = db_save(
                            db=db,
                            image_path=st.session_state.get("prediction_name", "upload"),
                            stage=stage,
                            confidence=confidence,
                            estimated_days_left=shelf_life.display(),
                            model_version=settings.model_version,
                            temperature_c=temperature,
                            humidity_pct=humidity,
                            storage_condition=storage_condition,
                            input_method=st.session_state.get("prediction_input_method", "upload"),
                            estimated_min_days=shelf_life.estimated_min_days,
                            estimated_max_days=shelf_life.estimated_max_days,
                            shelf_life_method=shelf_life.method,
                        )
                        st.session_state["saved_prediction_path"] = st.session_state["prediction_path"]
                        st.session_state["last_saved_prediction_id"] = row.id
                        st.success(f"Prediction saved (ID: {row.id}).")
                    except Exception as exc:
                        st.warning(f"Could not save: {exc}")
                    finally:
                        db.close()

        # FEATURE 12 -- Human feedback
        st.subheader("Feedback")
        if not st.session_state.get("feedback_given"):
            st.markdown("**Was this prediction correct?**")
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                if st.button("Yes, correct", key="fb_yes", use_container_width=True):
                    _save_feedback_to_db("correct", None)
            with fb_col2:
                if st.button("No, incorrect", key="fb_no", use_container_width=True):
                    st.session_state["show_correction"] = True

            if st.session_state.get("show_correction"):
                corrected = st.selectbox(
                    "What was the actual stage?",
                    options=["unripe", "ripe", "overripe", "rotten", "unsure"],
                    key="correction_select",
                )
                if st.button("Submit correction", key="fb_submit_correction"):
                    _save_feedback_to_db("incorrect", corrected)
        else:
            st.success("Feedback recorded. Thank you!")


# ===========================================================================
# PAGE: Batch Analysis (FEATURE 13)
# ===========================================================================

def page_batch():
    st.title("Batch Analysis")
    st.caption(
        "Upload multiple banana images (5-20 recommended). "
        "Results are NOT saved automatically."
    )

    batch_files = st.file_uploader(
        "Upload banana images for batch analysis",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.write(f"**{len(batch_files)} file(s) selected.**")
        run_batch = st.button("Run Batch Analysis", type="primary", key="batch_run_btn")

        if run_batch or st.session_state.get("batch_results"):
            if run_batch:
                if not Path(settings.model_path).exists():
                    st.error(f"Model not found: {settings.model_path}")
                else:
                    batch_results = []
                    progress_bar = st.progress(0, text="Processing batch...")
                    model_obj, transform, classes, device = cached_model(settings.model_path)

                    for i, bf in enumerate(batch_files):
                        try:
                            data = bf.getvalue()
                            img = validate_image_bytes(data, filename=bf.name)

                            # --- Banana content gate ---
                            banana_res = validate_banana_image(img)
                            if banana_res.state == ValidationState.NOT_BANANA:
                                batch_results.append({
                                    "filename": bf.name,
                                    "stage": "rejected",
                                    "confidence": 0,
                                    "probabilities": {},
                                    "estimated_days_left": "N/A",
                                    "error": "Not a banana image",
                                    "banana_detection_confidence": banana_res.confidence,
                                    "banana_state": "NOT_BANANA",
                                })
                                continue
                            if banana_res.state == ValidationState.UNCERTAIN:
                                batch_results.append({
                                    "filename": bf.name,
                                    "stage": "uncertain",
                                    "confidence": 0,
                                    "probabilities": {},
                                    "estimated_days_left": "N/A",
                                    "error": "Banana detection uncertain",
                                    "banana_detection_confidence": banana_res.confidence,
                                    "banana_state": "UNCERTAIN",
                                })
                                continue
                            # --- banana confirmed ---

                            tmp_name = safe_temp_filename(bf.name)
                            tmp_dir = Path("reports")
                            tmp_dir.mkdir(exist_ok=True)
                            tmp_path = tmp_dir / tmp_name
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG")
                            tmp_path.write_bytes(buf.getvalue())

                            result = predict_image(
                                str(tmp_path), model_obj, transform, classes, device
                            )
                            sl = estimate_shelf_life(
                                result["stage"], temperature, humidity, storage_condition
                            )
                            result["filename"] = bf.name
                            result["estimated_days_left"] = sl.display()
                            result["banana_detection_confidence"] = banana_res.confidence
                            result["banana_state"] = "BANANA"
                            batch_results.append(result)
                            tmp_path.unlink(missing_ok=True)
                        except ImageValidationError as exc:
                            batch_results.append({
                                "filename": bf.name,
                                "stage": "error",
                                "confidence": 0,
                                "probabilities": {},
                                "estimated_days_left": "N/A",
                                "error": str(exc),
                            })
                        except Exception as exc:
                            batch_results.append({
                                "filename": bf.name,
                                "stage": "error",
                                "confidence": 0,
                                "probabilities": {},
                                "estimated_days_left": "N/A",
                                "error": str(exc),
                            })
                        progress_bar.progress(
                            (i + 1) / len(batch_files),
                            text=f"Processed {i + 1}/{len(batch_files)}",
                        )

                    st.session_state["batch_results"] = batch_results
                    progress_bar.empty()

            batch_results = st.session_state.get("batch_results", [])
            if batch_results:
                valid = [r for r in batch_results if r["stage"] not in {"error", "rejected", "uncertain"}]
                rejected = [r for r in batch_results if r["stage"] == "rejected"]
                uncertain = [r for r in batch_results if r["stage"] == "uncertain"]
                errors = [r for r in batch_results if r["stage"] == "error"]

                stage_counts: dict[str, int] = {}
                for r in valid:
                    stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

                st.subheader("Batch Summary")
                sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
                sc1.metric("Total", len(batch_files))
                sc2.metric("Unripe", stage_counts.get("unripe", 0))
                sc3.metric("Ripe", stage_counts.get("ripe", 0))
                sc4.metric("Overripe", stage_counts.get("overripe", 0))
                sc5.metric("Rotten", stage_counts.get("rotten", 0))
                sc6.metric("Rejected", len(rejected))
                sc7.metric("Uncertain", len(uncertain))

                if rejected or uncertain:
                    st.info(
                        f"{len(rejected)} image(s) rejected (not a banana). "
                        f"{len(uncertain)} image(s) uncertain (banana not confirmed)."
                    )

                if errors:
                    st.warning(f"{len(errors)} file(s) could not be processed.")

                st.subheader("Individual Results")
                import pandas as pd

                # Show valid (banana) predictions
                rows = [
                    {
                        "Filename": r["filename"],
                        "Stage": r["stage"].capitalize(),
                        "Ripeness Confidence": f"{r['confidence']:.1%}",
                        "Banana Detection": f"{r.get('banana_detection_confidence', 0):.0%}",
                        "Est. Window": r["estimated_days_left"],
                        "Status": "OK",
                    }
                    for r in valid
                ]
                # Show rejected/uncertain in the same table
                for r in rejected:
                    rows.append({
                        "Filename": r["filename"],
                        "Stage": "Rejected",
                        "Ripeness Confidence": "N/A",
                        "Banana Detection": f"{r.get('banana_detection_confidence', 0):.0%}",
                        "Est. Window": "N/A",
                        "Status": "Image rejected (not a banana)",
                    })
                for r in uncertain:
                    rows.append({
                        "Filename": r["filename"],
                        "Stage": "Uncertain",
                        "Ripeness Confidence": "N/A",
                        "Banana Detection": f"{r.get('banana_detection_confidence', 0):.0%}",
                        "Est. Window": "N/A",
                        "Status": "Banana detection uncertain",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                csv_data = generate_batch_csv(batch_results)
                st.download_button(
                    "Export Batch CSV",
                    data=csv_data,
                    file_name="banana_ai_batch_results.csv",
                    mime="text/csv",
                    key="batch_csv_dl",
                )

                if st.button("Save Batch Results to Database", key="batch_save_btn"):
                    db = _get_db()
                    if db is None:
                        st.warning("PostgreSQL unavailable.")
                    else:
                        try:
                            from banana_ai.db.crud import save_prediction as db_save
                            saved = 0
                            # Only save confirmed banana predictions
                            for r in valid:
                                sl = estimate_shelf_life(
                                    r["stage"], temperature, humidity, storage_condition
                                )
                                db_save(
                                    db=db,
                                    image_path=r["filename"],
                                    stage=r["stage"],
                                    confidence=r["confidence"],
                                    estimated_days_left=sl.display(),
                                    model_version=settings.model_version,
                                    temperature_c=temperature,
                                    humidity_pct=humidity,
                                    storage_condition=storage_condition,
                                    input_method="upload",
                                    estimated_min_days=sl.estimated_min_days,
                                    estimated_max_days=sl.estimated_max_days,
                                    shelf_life_method=sl.method,
                                )
                                saved += 1
                            skipped = len(rejected) + len(uncertain)
                            msg = f"{saved} banana result(s) saved to database."
                            if skipped:
                                msg += f" {skipped} rejected/uncertain image(s) not saved."
                            st.success(msg)
                        except Exception as exc:
                            st.warning(f"Could not save batch: {exc}")
                        finally:
                            db.close()


# ===========================================================================
# PAGE: History (FEATURE 9)
# ===========================================================================

def page_history():
    st.title("Prediction History")
    st.caption("All saved predictions from PostgreSQL. Correlation does not imply causation.")

    db = _get_db()
    if db is None:
        st.warning("PostgreSQL unavailable. History requires a database connection.")
        return

    try:
        import pandas as pd
        from banana_ai.db.crud import all_predictions, total_predictions, count_by_stage

        total = total_predictions(db)
        stage_dist = count_by_stage(db)

        st.subheader("Overview")
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        oc1.metric("Total Scans", total)
        oc2.metric("Unripe", stage_dist.get("unripe", 0))
        oc3.metric("Ripe", stage_dist.get("ripe", 0))
        oc4.metric("Overripe", stage_dist.get("overripe", 0))
        oc5.metric("Rotten", stage_dist.get("rotten", 0))

        st.divider()
        st.subheader("Filters")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            stage_filter = st.selectbox(
                "Filter by stage",
                options=["All", "unripe", "ripe", "overripe", "rotten"],
                key="hist_stage_filter",
            )
        with fc2:
            min_conf = st.slider(
                "Min confidence", 0.0, 1.0, 0.0, 0.05, key="hist_conf_filter"
            )
        with fc3:
            model_filter = st.text_input("Model version (optional)", key="hist_model_filter")

        rows = all_predictions(
            db,
            stage_filter=None if stage_filter == "All" else stage_filter,
            min_confidence=min_conf if min_conf > 0 else None,
            model_version_filter=model_filter or None,
            limit=200,
        )

        table_data = [
            {
                "ID": pred.id,
                "Date": str(pred.created_at)[:19] if pred.created_at else "",
                "Stage": pred.predicted_stage.capitalize(),
                "Confidence": f"{pred.confidence:.1%}",
                "Temp (C)": obs.temperature_c,
                "Humidity (%)": obs.humidity_pct,
                "Est. Window": pred.estimated_days_left,
                "Input": getattr(obs, "input_method", "") or "",
                "Model": pred.model_version,
            }
            for pred, obs in rows
        ]

        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            st.caption(
                "Correlation between environmental conditions and predictions "
                "does not imply causation."
            )
        else:
            st.info("No predictions found matching the current filters.")
    except Exception as exc:
        st.warning(f"Could not load history: {exc}")
    finally:
        db.close()


# ===========================================================================
# PAGE: Analytics (FEATURE 10)
# ===========================================================================

def page_analytics():
    st.title("Analytics Dashboard")
    st.caption("Charts based on real stored data only. No fabricated data.")

    db = _get_db()
    if db is None:
        st.warning("PostgreSQL unavailable.")
        return

    try:
        import pandas as pd
        from banana_ai.db.crud import all_predictions, total_predictions, count_by_stage

        total = total_predictions(db)
        if total < 3:
            st.info(
                "Not enough historical data yet. "
                "Collect more scans to unlock analytics. "
                f"(Current: {total} record(s))"
            )
            return

        rows = all_predictions(db, limit=500)
        records = [
            {
                "date": pred.created_at,
                "stage": pred.predicted_stage,
                "confidence": pred.confidence,
                "temperature": obs.temperature_c,
                "humidity": obs.humidity_pct,
                "estimated_min": pred.estimated_min_days,
            }
            for pred, obs in rows
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        stage_dist = count_by_stage(db)
        ripe_count = stage_dist.get("ripe", 0)
        avg_confidence = float(df["confidence"].mean()) if not df.empty else 0.0
        metric_cols = st.columns(4)
        for col, label, value in [
            (metric_cols[0], "Total analyses", total),
            (metric_cols[1], "Bananas detected", total),
            (metric_cols[2], "Average AI confidence", f"{avg_confidence:.1%}"),
            (metric_cols[3], "Ripe", ripe_count),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.subheader("Ripeness Distribution")
        st.bar_chart(stage_dist)

        st.subheader("Confidence Distribution")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(df["confidence"], bins=20, color="#4caf50", edgecolor="white")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Count")
            ax.set_title("Model Confidence Distribution")
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.bar_chart(df["confidence"])

        st.subheader("Scans Over Time")
        if "date" in df.columns and df["date"].notna().any():
            scans_by_date = df.set_index("date").resample("D")["stage"].count()
            st.line_chart(scans_by_date)
        else:
            st.info("Date information unavailable for timeline chart.")

        if df["temperature"].notna().any():
            st.subheader("Temperature vs Stage")
            temp_stage = (
                df[df["temperature"].notna()]
                .groupby("stage")["temperature"]
                .mean()
            )
            st.bar_chart(temp_stage)
            st.caption("Correlation does not imply causation.")

    except Exception as exc:
        st.warning(f"Analytics unavailable: {exc}")
    finally:
        db.close()


# ===========================================================================
# PAGE: Waste Insights (FEATURE 14)
# ===========================================================================

def page_waste_insights():
    st.title("Banana Waste Insights")
    st.caption("Based on real stored data only. No fabricated environmental impact numbers.")

    db = _get_db()
    if db is None:
        st.warning("PostgreSQL unavailable.")
        return

    try:
        from banana_ai.db.crud import total_predictions, count_by_stage

        total = total_predictions(db)
        if total < 5:
            st.info(
                f"Collect more scans to unlock waste insights. "
                f"(Current: {total} record(s))"
            )
            return

        stage_dist = count_by_stage(db)
        st.subheader("BANANA WASTE INSIGHTS")
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        wc1.metric("Total Analyzed", total)
        wc2.metric("Unripe", stage_dist.get("unripe", 0))
        wc3.metric("Ripe", stage_dist.get("ripe", 0))
        wc4.metric("Overripe", stage_dist.get("overripe", 0))
        wc5.metric("Rotten", stage_dist.get("rotten", 0))

        rotten = stage_dist.get("rotten", 0)
        overripe = stage_dist.get("overripe", 0)
        at_risk = rotten + overripe
        at_risk_pct = (at_risk / total * 100) if total > 0 else 0

        st.info(
            f"{at_risk_pct:.1f}% of analyzed bananas were overripe or rotten "
            f"({at_risk} of {total}). These are descriptive statistics only -- "
            "no weight or environmental impact claims are made."
        )
        st.caption(
            "These statistics describe the bananas analyzed by this system. "
            "They do not represent a population sample or imply food-waste impact claims."
        )
    except Exception as exc:
        st.warning(f"Waste insights unavailable: {exc}")
    finally:
        db.close()


# ===========================================================================
# PAGE: About AI (FEATURE 21)
# ===========================================================================

def page_about_ai():
    st.title("About the AI")

    st.markdown(f"""
## Banana Gate Binary Classifier -- TRAINED AI

| Property | Value |
|---|---|
| Model Checkpoint | `models/banana_gate_best.pt` |
| Architecture | `BananaGateMobileNetV3` (MobileNetV3-Small transfer learning) |
| Purpose | Dedicated Gate: Banana vs. Non-Banana |
| Checkpoint SHA-256 | `a1416daae0ae22080af79436ae8c97b59ebcd301af28fddb8ce19dff5929789d` |
| Evaluation operating point | **0.380** (selected strictly on validation split) |
| Application acceptance threshold | **{settings.banana_gate_threshold:.3f}** (conservative default) |
| Test Accuracy | **99.67%** (600 held-out test images) |
| Banana Recall (TPR) | **100.00%** (300/300 bananas accepted) |
| Non-Banana Rejection (TNR) | **99.33%** (298/300 non-bananas blocked) |
| Test F1 Score | **0.9967** |
| Inference Latency | **< 15ms** on CPU |

> The gate stops non-banana inputs immediately. Non-banana images never reach the ripeness CNN (`banana_cnn_v2.pt`), Grad-CAM, shelf-life estimation, or database prediction storage.

---

## Banana Ripeness Classifier -- TRAINED AI

| Property | Value |
|---|---|
| Model | `banana-cnn-v2` |
| Architecture | Custom CNN trained from scratch |
| Classes | `overripe`, `ripe`, `rotten`, `unripe` |
| Parameters | ~422,788 |
| Validation macro-F1 | **0.9608** |
| Test accuracy | **95.20%** |
| Test macro-F1 | **0.9536** |

**Per-class test F1:**
- Overripe: 0.938
- Ripe: 0.942
- Rotten: 0.948
- Unripe: 0.986

> These are offline evaluation results on the held-out test set.
> They do NOT guarantee real-world performance on all images.

---

## Shelf-Life Estimator -- PROTOTYPE HEURISTIC

The good-quality window estimate is **NOT** a trained model.
It uses a simple bounded formula with prototype coefficients:

- **Baseline ranges** (days): unripe 4-7, ripe 2-4, overripe 0-2, rotten 0
- **Temperature adjustment**: ~0.08 days per deg C deviation from 22 deg C reference
- **Humidity adjustment**: ~0.05 days per % deviation from 60% reference
- **Storage multiplier**: room x1.0, cool x1.3, refrigerator x1.6

All coefficients are **prototype assumptions, not experimentally validated constants**.

---

## Future Trained Shelf-Life Model

To replace the heuristic with a real ML model, collect a **longitudinal dataset**:

| Field | Description |
|---|---|
| `banana_id` | Unique identifier per banana bunch |
| `image_path` | Captured image |
| `temperature_c` | Ambient temperature |
| `humidity_pct` | Relative humidity |
| `storage_condition` | Storage type |
| `days_since_start` | Days since observation started |
| `days_left` | Days until quality threshold crossed |
| `observed_stage` | Stage label |

**Important:** Split longitudinal data by `banana_id` to prevent data leakage.

---

## Limitations and Honest Disclaimers

- Model confidence is a **raw softmax score**, not a calibrated probability.
- Grad-CAM is a **visualization tool**, not proof of the model's reasoning.
- The shelf-life estimate is a **prototype heuristic** and NOT a food-safety guarantee.
- The system **cannot** determine whether a banana is safe to eat.
- Always inspect for visible mold, unusual odor, leakage, or other spoilage signs.
    """)


# ===========================================================================
# MAIN ROUTER
# ===========================================================================

if page == PAGE_ANALYZE:
    page_analyze()
elif page == PAGE_BATCH:
    page_batch()
elif page == PAGE_HISTORY:
    page_history()
elif page == PAGE_ANALYTICS:
    page_analytics()
elif page == PAGE_WASTE:
    page_waste_insights()
elif page == PAGE_ABOUT:
    page_about_ai()
