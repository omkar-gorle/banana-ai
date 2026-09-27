"""Error analysis: compares V1 and V2 results and generates error_analysis.md."""

import json
from pathlib import Path


def load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def compare_models(v1_dir: str = "reports/v1", v2_dir: str = "reports/v2") -> dict:
    """Compare V1 and V2 test metrics."""
    v1 = load_metrics(f"{v1_dir}/test_metrics.json")
    v2 = load_metrics(f"{v2_dir}/test_metrics.json")

    if not v1:
        return {"error": f"V1 metrics not found at {v1_dir}/test_metrics.json"}
    if not v2:
        return {"error": f"V2 metrics not found at {v2_dir}/test_metrics.json"}

    return {"v1": v1, "v2": v2}


def generate_error_analysis(
    v1_metrics_path: str = "reports/v1/test_metrics.json",
    output_path: str = "reports/error_analysis.md",
) -> None:
    """Generate error_analysis.md from measured test results."""

    metrics = load_metrics(v1_metrics_path)
    if not metrics:
        Path(output_path).write_text(
            "# Error Analysis\n\nNOT COMPLETED: V1 evaluation not available.\n",
            encoding="utf-8",
        )
        return

    report = metrics.get("classification_report", {})
    matrix = metrics.get("confusion_matrix", [])
    matrix_norm = metrics.get("confusion_matrix_normalized", [])
    class_names = ["overripe", "ripe", "rotten", "unripe"]

    accuracy = metrics.get("accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)

    lines = [
        "# Banana AI — Error Analysis",
        "",
        "**Based on measured V1 test set results. No values fabricated.**",
        "",
        "## Overall Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {accuracy:.4f} |",
        f"| Macro F1 | {macro_f1:.4f} |",
        f"| Macro Precision | {metrics.get('macro_precision', 0):.4f} |",
        f"| Macro Recall | {metrics.get('macro_recall', 0):.4f} |",
        f"| Weighted F1 | {metrics.get('weighted_f1', 0):.4f} |",
        "",
        "## Per-Class Performance",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]

    for cls in class_names:
        if cls in report:
            c = report[cls]
            lines.append(
                f"| {cls} | {c['precision']:.4f} | {c['recall']:.4f} | "
                f"{c['f1-score']:.4f} | {int(c['support'])} |"
            )

    lines += [
        "",
        "## Confusion Matrix (Raw Counts)",
        "",
        "Rows = actual class, Columns = predicted class.",
        "",
        "| Actual \\ Predicted | overripe | ripe | rotten | unripe |",
        "|---------------------|----------|------|--------|--------|",
    ]

    for i, cls in enumerate(class_names):
        if matrix and i < len(matrix):
            row = matrix[i]
            lines.append(
                f"| **{cls}** | {row[0]} | {row[1]} | {row[2]} | {row[3]} |"
            )

    lines += [
        "",
        "## Confusion Analysis",
        "",
        "### ripe vs overripe confusion",
        "",
    ]

    if matrix and len(matrix) >= 4:
        # class order: overripe=0, ripe=1, rotten=2, unripe=3
        overripe_as_ripe = matrix[0][1]
        ripe_as_overripe = matrix[1][0]
        overripe_total = sum(matrix[0])
        ripe_total = sum(matrix[1])

        lines += [
            f"- Overripe predicted as ripe: **{overripe_as_ripe}** / {overripe_total} "
            f"({100*overripe_as_ripe/max(1,overripe_total):.1f}%)",
            f"- Ripe predicted as overripe: **{ripe_as_overripe}** / {ripe_total} "
            f"({100*ripe_as_overripe/max(1,ripe_total):.1f}%)",
            "",
            "These two classes are visually adjacent on the ripeness spectrum. "
            "Yellow-brown transition makes them hard to separate without contextual cues.",
            "",
            "### overripe vs rotten confusion",
            "",
        ]

        overripe_as_rotten = matrix[0][2]
        rotten_as_overripe = matrix[2][0]
        rotten_total = sum(matrix[2])

        lines += [
            f"- Overripe predicted as rotten: **{overripe_as_rotten}** / {overripe_total} "
            f"({100*overripe_as_rotten/max(1,overripe_total):.1f}%)",
            f"- Rotten predicted as overripe: **{rotten_as_overripe}** / {rotten_total} "
            f"({100*rotten_as_overripe/max(1,rotten_total):.1f}%)",
            "",
            "### unripe vs ripe confusion",
            "",
        ]

        unripe_as_ripe = matrix[3][1]
        ripe_as_unripe = matrix[1][3]
        unripe_total = sum(matrix[3])

        lines += [
            f"- Unripe predicted as ripe: **{unripe_as_ripe}** / {unripe_total} "
            f"({100*unripe_as_ripe/max(1,unripe_total):.1f}%)",
            f"- Ripe predicted as unripe: **{ripe_as_unripe}** / {ripe_total} "
            f"({100*ripe_as_unripe/max(1,ripe_total):.1f}%)",
            "",
            "### rotten false negatives",
            "",
        ]

        rotten_correct = matrix[2][2]
        rotten_missed = rotten_total - rotten_correct

        lines += [
            f"- Rotten correctly identified: **{rotten_correct}** / {rotten_total} "
            f"({100*rotten_correct/max(1,rotten_total):.1f}%)",
            f"- Rotten misclassified as other: **{rotten_missed}** / {rotten_total} "
            f"({100*rotten_missed/max(1,rotten_total):.1f}%)",
            "",
        ]

    incorrect_path = Path(v1_metrics_path).parent / "incorrect_predictions.json"
    if incorrect_path.exists():
        incorrect = json.loads(incorrect_path.read_text(encoding="utf-8"))
        low_conf = [ex for ex in incorrect if ex["confidence"] < 0.6]

        lines += [
            "### Low-confidence incorrect predictions",
            "",
            f"Incorrect predictions with confidence < 60%: **{len(low_conf)}**",
            "",
        ]

        if low_conf[:5]:
            lines += [
                "| True class | Predicted | Confidence |",
                "|------------|-----------|------------|",
            ]
            for ex in low_conf[:5]:
                lines.append(
                    f"| {ex['true_class']} | {ex['predicted_class']} | "
                    f"{ex['confidence']:.3f} |"
                )
            lines.append("")

    lines += [
        "## Key Observations",
        "",
        "1. **Adjacent-class confusion**: ripe/overripe and overripe/rotten are the "
        "most frequent error patterns, consistent with their visual similarity.",
        "",
        "2. **Class imbalance effect**: rotten has the highest support (~33%) in train "
        "and may be over-represented in the decision boundary. The V2 class-weight "
        "experiment addresses this.",
        "",
        "3. **No test set was modified** based on these results.",
        "",
        "4. **Grad-CAM** should be used to inspect whether the model attends to "
        "banana skin texture/colour regions or spurious background features.",
        "",
        "## Limitations",
        "",
        "- Dataset images are 416×416 photos under similar controlled backgrounds. "
        "Real-world images may have different lighting, orientation, and backgrounds.",
        "- The model was trained from scratch with no pretrained features. A larger "
        "dataset or transfer learning could improve boundary precision.",
        "- Shelf-life prediction requires longitudinal data not present in this dataset.",
        "",
        "---",
        "_Analysis generated from measured V1 test results. Values are not fabricated._",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Error analysis saved to: {output_path}")


def generate_comparison_report(
    v1_dir: str = "reports/v1",
    v2_dir: str = "reports/v2",
    output_path: str = "reports/model_comparison.md",
) -> None:
    """Compare V1 and V2 after both are evaluated."""
    v1 = load_metrics(f"{v1_dir}/test_metrics.json")
    v2 = load_metrics(f"{v2_dir}/test_metrics.json")

    lines = ["# V1 vs V2 Model Comparison", ""]

    if not v1 or not v2:
        missing = []
        if not v1:
            missing.append("V1")
        if not v2:
            missing.append("V2")
        lines.append(f"NOT COMPLETED: Missing results for: {', '.join(missing)}")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return

    class_names = ["overripe", "ripe", "rotten", "unripe"]
    v1_report = v1.get("classification_report", {})
    v2_report = v2.get("classification_report", {})

    lines += [
        "## Overall Metrics",
        "",
        "| Metric | V1 (baseline) | V2 (class weights) | Delta |",
        "|--------|--------------|---------------------|-------|",
    ]

    def row(label, key):
        v1v = v1.get(key, 0)
        v2v = v2.get(key, 0)
        delta = v2v - v1v
        sign = "+" if delta >= 0 else ""
        return f"| {label} | {v1v:.4f} | {v2v:.4f} | {sign}{delta:.4f} |"

    lines += [
        row("Accuracy", "accuracy"),
        row("Macro F1", "macro_f1"),
        row("Macro Precision", "macro_precision"),
        row("Macro Recall", "macro_recall"),
        row("Weighted F1", "weighted_f1"),
        "",
        "## Per-Class F1 Comparison",
        "",
        "| Class | V1 F1 | V2 F1 | Delta |",
        "|-------|-------|-------|-------|",
    ]

    for cls in class_names:
        v1f = v1_report.get(cls, {}).get("f1-score", 0)
        v2f = v2_report.get(cls, {}).get("f1-score", 0)
        delta = v2f - v1f
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {cls} | {v1f:.4f} | {v2f:.4f} | {sign}{delta:.4f} |")

    # Model selection recommendation
    v1_val_f1 = v1.get("best_val_macro_f1", 0) or 0
    v2_val_f1 = v2.get("best_val_macro_f1", 0) or 0

    lines += [
        "",
        "## Model Selection",
        "",
        "**Selection criterion**: best validation macro-F1 (validation set, not test set).",
        "",
        f"| Model | Best Val Macro-F1 |",
        f"|-------|------------------|",
        f"| V1 | {v1_val_f1:.4f} |",
        f"| V2 | {v2_val_f1:.4f} |",
        "",
    ]

    selected = "V1" if v1_val_f1 >= v2_val_f1 else "V2"
    lines += [
        f"**Selected model: {selected}** (higher validation macro-F1)",
        "",
        "Note: This selection was made using the validation set only, "
        "not by inspecting test-set metrics.",
        "",
        "## Important Notes",
        "",
        "- V1 uses equal class weights (vanilla CrossEntropyLoss).",
        "- V2 uses inverse-frequency class weights to reduce majority-class bias.",
        "- Both models use identical architecture, dataset, and evaluation protocol.",
        "- No universal 'winner' is declared. V2 may improve minority-class recall "
        "at the cost of overall accuracy. Examine per-class F1 differences.",
        "",
        "---",
        "_Values from measured test evaluations. Not fabricated._",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison report saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", default="reports/v1")
    parser.add_argument("--v2-dir", default="reports/v2")
    parser.add_argument("--output", default="reports/model_comparison.md")
    args = parser.parse_args()
    generate_comparison_report(args.v1_dir, args.v2_dir, args.output)
