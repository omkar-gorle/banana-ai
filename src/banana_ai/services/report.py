"""Report generation service for Banana AI.

Generates downloadable scan reports in CSV and HTML formats.
PDF generation is NOT implemented to avoid adding large optional
dependencies (e.g. weasyprint/reportlab) — HTML can be saved/printed
as PDF from the browser.

All reports include the mandatory disclaimer:
"This report is an AI-assisted assessment. The shelf-life estimate is a
prototype heuristic and is not a food-safety guarantee."
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Optional


_DISCLAIMER = (
    "This report is an AI-assisted assessment. "
    "The shelf-life estimate is a prototype heuristic and is NOT a food-safety guarantee. "
    "Check the fruit for visible mold, unusual odor, leakage, or other signs of spoilage. "
    "Model confidence is a raw softmax score and does not represent calibrated probability."
)


def generate_scan_report_html(
    prediction: dict[str, Any],
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    storage_condition: Optional[str] = None,
    shelf_life: Optional[Any] = None,
    input_method: str = "upload",
    model_version: str = "banana-cnn-v2",
) -> str:
    """Generate an HTML scan report.

    Parameters
    ----------
    prediction : dict
        Result dict from ``predict_image`` with keys:
        ``stage``, ``confidence``, ``probabilities``, ``estimated_days_left``.
    temperature_c : float, optional
    humidity_pct : float, optional
    storage_condition : str, optional
    shelf_life : ShelfLifeEstimate, optional
        Full shelf-life estimate object.
    input_method : str
        ``"camera"`` or ``"upload"``.
    model_version : str

    Returns
    -------
    str
        HTML report as a string. Suitable for st.download_button.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage = prediction.get("stage", "unknown").upper()
    confidence = prediction.get("confidence", 0.0)
    probabilities = prediction.get("probabilities", {})
    estimated_days = prediction.get("estimated_days_left", "N/A")

    sl_min = getattr(shelf_life, "estimated_min_days", None)
    sl_max = getattr(shelf_life, "estimated_max_days", None)
    sl_explanation = getattr(shelf_life, "explanation", "")
    sl_display = getattr(shelf_life, "display", lambda: estimated_days)()

    prob_rows = "".join(
        f"<tr><td>{k.capitalize()}</td><td>{v:.1%}</td>"
        f'<td><div style="background:#4caf50;width:{v*200:.0f}px;height:14px;border-radius:3px"></div></td></tr>'
        for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    env_section = ""
    if temperature_c is not None or humidity_pct is not None:
        env_section = f"""
        <h2>🌡️ Environmental Conditions</h2>
        <table>
          <tr><td>Temperature</td><td>{temperature_c if temperature_c is not None else 'Not provided'} °C</td></tr>
          <tr><td>Humidity</td><td>{humidity_pct if humidity_pct is not None else 'Not provided'} %</td></tr>
          <tr><td>Storage Condition</td><td>{storage_condition or 'Not specified'}</td></tr>
        </table>
        <h2>⏳ Estimated Good-Quality Window</h2>
        <p><strong>~{sl_display}</strong> (prototype heuristic — not a food-safety guarantee)</p>
        <p>{sl_explanation}</p>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Banana AI Scan Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; color: #222; }}
  h1 {{ color: #e65100; }}
  h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
  td, th {{ border: 1px solid #ddd; padding: 8px 12px; }}
  th {{ background: #f5f5f5; }}
  .disclaimer {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 12px; margin-top: 24px; font-size: 0.9em; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:12px; font-weight:bold; color:#fff; }}
  .ripe {{ background:#4caf50; }}
  .overripe {{ background:#ff9800; }}
  .rotten {{ background:#f44336; }}
  .unripe {{ background:#2196f3; }}
</style>
</head>
<body>
<h1>🍌 Banana AI Scan Report</h1>
<p><strong>Generated:</strong> {now}</p>
<p><strong>Input method:</strong> {input_method.capitalize()}</p>
<p><strong>Model:</strong> {model_version}</p>

<h2>🍌 Prediction</h2>
<p>Predicted Stage: <span class="badge {stage.lower()}">{stage}</span></p>
<p>Model Confidence: <strong>{confidence:.1%}</strong>
   <em>(raw softmax score — not a calibrated probability)</em></p>

<h2>📊 Class Probabilities</h2>
<table>
  <tr><th>Stage</th><th>Score</th><th>Bar</th></tr>
  {prob_rows}
</table>

{env_section}

<h2>🤖 About the Model</h2>
<table>
  <tr><td>Model</td><td>{model_version}</td></tr>
  <tr><td>Architecture</td><td>Custom CNN trained from scratch</td></tr>
  <tr><td>Classes</td><td>overripe, ripe, rotten, unripe</td></tr>
  <tr><td>Validation macro-F1</td><td>0.9608</td></tr>
  <tr><td>Test accuracy</td><td>95.20%</td></tr>
  <tr><td>Test macro-F1</td><td>0.9536</td></tr>
</table>
<p><em>These are offline evaluation results on the held-out test set.
They do not guarantee real-world performance on all images.</em></p>

<div class="disclaimer">
  <strong>⚠️ Important Limitations</strong><br>
  {_DISCLAIMER}
</div>
</body>
</html>"""


def generate_scan_report_csv(
    prediction: dict[str, Any],
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    storage_condition: Optional[str] = None,
    shelf_life: Optional[Any] = None,
    input_method: str = "upload",
    model_version: str = "banana-cnn-v2",
) -> str:
    """Generate a CSV scan report as a string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage = prediction.get("stage", "unknown")
    confidence = prediction.get("confidence", 0.0)
    probabilities = prediction.get("probabilities", {})
    sl_display = getattr(shelf_life, "display", lambda: prediction.get("estimated_days_left", "N/A"))()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Field", "Value"])
    writer.writerow(["Generated", now])
    writer.writerow(["Input Method", input_method])
    writer.writerow(["Model Version", model_version])
    writer.writerow(["Predicted Stage", stage])
    writer.writerow(["Model Confidence", f"{confidence:.4f}"])
    writer.writerow(["Temperature (°C)", temperature_c if temperature_c is not None else ""])
    writer.writerow(["Humidity (%)", humidity_pct if humidity_pct is not None else ""])
    writer.writerow(["Storage Condition", storage_condition or ""])
    writer.writerow(["Estimated Good-Quality Window", sl_display])
    writer.writerow([])
    writer.writerow(["Class Probabilities", ""])
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([cls.capitalize(), f"{prob:.4f}"])
    writer.writerow([])
    writer.writerow(["Disclaimer", _DISCLAIMER])

    return output.getvalue()


def generate_batch_csv(batch_results: list[dict[str, Any]]) -> str:
    """Generate a CSV summary of batch analysis results."""
    if not batch_results:
        return ""

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Filename", "Predicted Stage", "Confidence",
        "Overripe %", "Ripe %", "Rotten %", "Unripe %",
        "Estimated Window"
    ]
    writer.writerow(headers)

    for r in batch_results:
        probs = r.get("probabilities", {})
        writer.writerow([
            r.get("filename", ""),
            r.get("stage", ""),
            f"{r.get('confidence', 0):.4f}",
            f"{probs.get('overripe', 0):.4f}",
            f"{probs.get('ripe', 0):.4f}",
            f"{probs.get('rotten', 0):.4f}",
            f"{probs.get('unripe', 0):.4f}",
            r.get("estimated_days_left", ""),
        ])

    return output.getvalue()
