import json
from pathlib import Path

hist_path = Path("reports/training_history.json")
if hist_path.exists():
    hist = json.loads(hist_path.read_text())
    print(f"Total epochs: {len(hist)}")
    for r in hist:
        ep = r["epoch"]
        tl = r["train_loss"]
        ta = r["train_accuracy"]
        tf = r["train_macro_f1"]
        vl = r["val_loss"]
        va = r["val_accuracy"]
        vf = r["val_macro_f1"]
        lr = r["learning_rate"]
        dur = r["epoch_duration_seconds"]
        print(
            f"Ep{ep:02d}: train_loss={tl:.4f} train_acc={ta:.4f} train_f1={tf:.4f}"
            f" | val_loss={vl:.4f} val_acc={va:.4f} val_f1={vf:.4f}"
            f" | lr={lr:.2e} | dur={dur:.0f}s"
        )
else:
    print("training_history.json NOT FOUND")
