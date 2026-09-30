import json
import subprocess
from pathlib import Path

def evaluate(model_path, dataset_root, output_dir, label):
    cmd = [
        ".venv/Scripts/python.exe", "src/banana_ai/ml/evaluate_extended.py",
        "--checkpoint", model_path,
        "--dataset-root", dataset_root,
        "--output-dir", output_dir,
        "--version-label", label
    ]
    print(f"Running: {' '.join(cmd)}")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, check=True, env=env)

def analyze_errors(incorrect_path, total_predictions):
    if not Path(incorrect_path).exists():
        return {}
    with open(incorrect_path) as f:
        incorrect = json.load(f)
        
    stats = {
        "count_90": 0, "count_95": 0, "count_99": 0,
        "total_errors": len(incorrect),
        "total_preds": total_predictions,
        "dangerous": 0, "conservative": 0,
        "examples": []
    }
    
    for err in incorrect:
        conf = err["confidence"]
        if conf >= 0.90: stats["count_90"] += 1
        if conf >= 0.95: stats["count_95"] += 1
        if conf >= 0.99: stats["count_99"] += 1
        
        # Shelf-life safety
        true_label = err["true_class"]
        pred_label = err["predicted_class"]
        if true_label == "rotten" and pred_label in ["ripe", "unripe"]:
            stats["dangerous"] += 1
        if true_label in ["ripe", "unripe"] and pred_label == "rotten":
            stats["conservative"] += 1
            
        if conf >= 0.99 and len(stats["examples"]) < 5:
            stats["examples"].append(err)
            
    return stats

def main():
    # 1. Run evaluations on original test set
    evaluate("models/banana_cnn_v2.pt", "D:/ff/banana/banana_classification", "reports/v2_orig", "v2_orig")
    evaluate("models/banana_cnn_v3.pt", "D:/ff/banana/banana_classification", "reports/v3_orig", "v3_orig")
    
    # We already have rw (expanded) evaluations from previous steps in reports/v2_baseline and reports/v3_evaluation
    # But just to be sure we'll run them to guarantee fresh data
    evaluate("models/banana_cnn_v2.pt", "D:/ff/banana/banana_classification_v3", "reports/v2_rw", "v2_rw")
    evaluate("models/banana_cnn_v3.pt", "D:/ff/banana/banana_classification_v3", "reports/v3_rw", "v3_rw")

    def get_metrics(dir_name):
        with open(f"{dir_name}/test_metrics.json") as f:
            return json.load(f)
            
    m_v2_orig = get_metrics("reports/v2_orig")
    m_v3_orig = get_metrics("reports/v3_orig")
    m_v2_rw = get_metrics("reports/v2_rw")
    m_v3_rw = get_metrics("reports/v3_rw")
    
    total_orig = m_v2_orig["classification_report"]["macro avg"]["support"]
    total_rw = m_v2_rw["classification_report"]["macro avg"]["support"]
    
    err_v2_orig = analyze_errors("reports/v2_orig/incorrect_predictions.json", total_orig)
    err_v3_orig = analyze_errors("reports/v3_orig/incorrect_predictions.json", total_orig)
    err_v2_rw = analyze_errors("reports/v2_rw/incorrect_predictions.json", total_rw)
    err_v3_rw = analyze_errors("reports/v3_rw/incorrect_predictions.json", total_rw)
    
    with open("reports/verification_results.json", "w") as f:
        json.dump({
            "metrics": {
                "v2_orig": m_v2_orig,
                "v3_orig": m_v3_orig,
                "v2_rw": m_v2_rw,
                "v3_rw": m_v3_rw
            },
            "errors": {
                "v2_orig": err_v2_orig,
                "v3_orig": err_v3_orig,
                "v2_rw": err_v2_rw,
                "v3_rw": err_v3_rw
            }
        }, f, indent=2)

if __name__ == "__main__":
    main()
