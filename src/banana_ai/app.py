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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.prediction-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    text-align: center;
    margin: 16px 0;
}
.stage-ripe    { color: #4caf50; font-size: 2rem; font-weight: 700; }
.stage-overripe { color: #ff9800; font-size: 2rem; font-weight: 700; }
.stage-rotten  { color: #f44336; font-size: 2rem; font-weight: 700; }
.stage-unripe  { color: #2196f3; font-size: 2rem; font-weight: 700; }
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


def _validate_and_save_temp(data: bytes, filename: str) -> Optional[Path]:
    """Validate image bytes and save to a temp file; return Path or None."""
    try:
        img = validate_image_bytes(data, filename=filename)
    except ImageValidationError as exc:
        st.error(f"Image rejected: {exc}")
        return None

    safe_name = safe_temp_filename(filename)
    tmp_dir = Path("reports")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / safe_name

    with tmp_path.open("wb") as f:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        f.write(img_bytes.getvalue())

    return tmp_path


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
    st.markdown("## Banana AI")
    st.caption("Quality & Shelf-Life Intelligence System")
    st.divider()

    st.markdown("### Model")
    st.code(settings.model_path, language=None)
    st.write(f"**Version:** `{settings.model_version}`")

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


# ===========================================================================
# PAGE: Analyze (FEATURES 1-12, 15-18)
# ===========================================================================

def page_analyze():
    st.title("Banana AI")
    st.caption(
        "Banana Quality & Shelf-Life Intelligence System | "
        f"Model: {settings.model_version}"
    )
    st.info(
        "Mobile users: Tap 'Take Photo' to use your device camera, "
        "or 'Upload Photo' to select an existing image."
    )

    # FEATURE 1 -- Dual input
    tab_camera, tab_upload = st.tabs(["Take Photo", "Upload Photo"])

    active_image_bytes: Optional[bytes] = None
    active_filename: str = ""
    active_input_method: str = "upload"

    with tab_camera:
        st.markdown("**Use your device camera to capture a banana.**")
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

    with tab_upload:
        st.markdown("**Upload an existing banana photo.**")
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

    # Image preview + analyze button
    if active_image_bytes:
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
                image_path = _validate_and_save_temp(active_image_bytes, active_filename)
                if image_path:
                    try:
                        result = _run_prediction(image_path)
                        if result:
                            st.session_state["prediction"] = result
                            st.session_state["prediction_path"] = str(image_path)
                            st.session_state["prediction_name"] = active_filename
                            st.session_state["prediction_input_method"] = active_input_method
                            st.session_state["saved_prediction_path"] = None
                            st.session_state["feedback_given"] = False
                    except (OSError, ValueError, RuntimeError) as exc:
                        st.error(f"Could not analyse image: {exc}")
                        logger.exception("Prediction error")

    # Results
    result = st.session_state.get("prediction")
    if result:
        stage = result["stage"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        input_method = st.session_state.get("prediction_input_method", "upload")

        st.divider()

        # FEATURE 2 -- Prediction result card
        st.markdown(f"""
        <div class="prediction-card">
            <h2>BANANA ANALYSIS</h2>
            <p class="stage-{stage}">{stage.upper()}</p>
            <p>Model confidence: <strong>{confidence:.1%}</strong></p>
            <p style="font-size:0.8rem;opacity:0.7">
                Model: {settings.model_version} | Input: {input_method.capitalize()}
            </p>
            <p style="font-size:0.75rem;opacity:0.6">
                Confidence is a raw softmax score, not a calibrated probability.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # FEATURE 3 -- Quality assessment
        assessment = _ASSESSMENT.get(stage, "Assessment unavailable.")
        st.info(f"**Assessment:** {assessment}")

        # FEATURE 15 -- Eat-first priority
        priority_label = _PRIORITY_LABELS.get(stage, "")
        if priority_label:
            st.markdown(f"**Eat-First Priority:** {priority_label}")

        # Class probabilities (FEATURE 2)
        with st.expander("All class probabilities", expanded=True):
            for name, prob in sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                st.progress(prob, text=f"{name.capitalize()}: {prob:.1%}")
            st.caption("Raw softmax scores. All four scores sum to ~100%.")

        # FEATURE 4 & 5 -- Environment-aware shelf-life
        st.divider()
        st.subheader("Estimated Good-Quality Window")

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
                st.metric("Estimated window", f"~{shelf_life.display()}")
            with col_sl2:
                st.metric(
                    "Environment",
                    f"{temperature:.0f} deg C / {humidity:.0f}%"
                )

            with st.expander("How was this estimated?", expanded=False):
                st.write(shelf_life.explanation)
                st.caption(
                    "Prototype Heuristic -- coefficients are prototype assumptions, "
                    "NOT experimentally validated scientific constants."
                )

        st.warning(shelf_life.warning)

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
                valid = [r for r in batch_results if r["stage"] != "error"]
                errors = [r for r in batch_results if r["stage"] == "error"]

                stage_counts: dict[str, int] = {}
                for r in valid:
                    stage_counts[r["stage"]] = stage_counts.get(r["stage"], 0) + 1

                st.subheader("Batch Summary")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Total", len(batch_files))
                sc2.metric("Unripe", stage_counts.get("unripe", 0))
                sc3.metric("Ripe", stage_counts.get("ripe", 0))
                sc4.metric("Overripe", stage_counts.get("overripe", 0))
                sc5.metric("Rotten", stage_counts.get("rotten", 0))

                if errors:
                    st.warning(f"{len(errors)} file(s) could not be processed.")

                st.subheader("Individual Results")
                import pandas as pd
                rows = [
                    {
                        "Filename": r["filename"],
                        "Stage": r["stage"].capitalize(),
                        "Confidence": f"{r['confidence']:.1%}",
                        "Est. Window": r["estimated_days_left"],
                    }
                    for r in valid
                ]
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
                            st.success(f"{saved} batch results saved to database.")
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

        st.subheader("Ripeness Distribution")
        stage_dist = count_by_stage(db)
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

    st.markdown("""
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
