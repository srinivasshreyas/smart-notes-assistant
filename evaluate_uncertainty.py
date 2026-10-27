"""Choose the 'uncertain' rule from the model's own probabilities.

One command: python evaluate_uncertainty.py        (writes model/config.json)

Method A (primary): uncertain when the highest class probability < threshold.
Method B (compared): a 5th class "Other" trained on the TUNING half of the out-of-scope notes;
                     uncertain when Other is the top class or max in-scope probability < threshold.
The 12 out-of-scope notes are split 6/6 with a fixed seed: a TUNING half (may be used to choose)
and a HELD-OUT half (only reported). validation.csv is only used to count false alarms.
Rule fixed in advance: maximise (tuning OOS flagged - validation flagged), rejecting any
threshold that flags more than 1/3 of validation notes.
"""
import json
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train import build_model

ROOT = Path(__file__).parent
SEED = 42
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.901, 0.05)]
MAX_VAL_FLAGGED = 1 / 3


def split_oos(oos: pd.DataFrame):
    idx = np.random.RandomState(SEED).permutation(len(oos))
    half = len(oos) // 2
    return oos.iloc[idx[:half]].reset_index(drop=True), oos.iloc[idx[half:]].reset_index(drop=True)


def flagged(maxp: np.ndarray, t: float) -> float:
    return float((maxp < t).mean())


def pick(rows):
    ok = [r for r in rows if r["val"] <= MAX_VAL_FLAGGED]
    return max(ok, key=lambda r: (r["tune"] - r["val"], -r["thr"])) if ok else None


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    val = pd.read_csv(ROOT / "data" / "validation.csv")
    oos = pd.read_csv(ROOT / "data" / "out_of_scope_examples.csv")
    tune, held = split_oos(oos)
    print("Tuning half  :", " | ".join(tune["text"]))
    print("Held-out half:", " | ".join(held["text"]), "\n")

    model = joblib.load(ROOT / "model" / "model.joblib")
    mv, mt, mh = (model.predict_proba(d["text"]).max(axis=1) for d in (val, tune, held))

    rows = [dict(thr=t, val=flagged(mv, t), tune=flagged(mt, t), held=flagged(mh, t)) for t in THRESHOLDS]
    print("Method A: uncertain if max probability < threshold")
    print(f"{'threshold':>9} {'val flagged':>12} {'tuning OOS':>11} {'held-out OOS':>13}")
    for r in rows:
        print(f"{r['thr']:>9.2f} {r['val']:>12.0%} {r['tune']:>11.0%} {r['held']:>13.0%}")
    best = pick(rows)
    print(f"\nMethod A chosen threshold (rule above): {best['thr']:.2f}  score={best['tune']-best['val']:.2f}")

    # Method B: extra 'Other' class, trained on the tuning half only.
    extra = pd.DataFrame({"text": tune["text"], "category": "Other"})
    modelB = build_model().fit(pd.concat([train, extra])["text"], pd.concat([train, extra])["category"])
    classes = list(modelB.classes_)
    oi = classes.index("Other")

    def flag_b(texts, t):
        p = modelB.predict_proba(texts)
        return float(((p.argmax(1) == oi) | (np.delete(p, oi, axis=1).max(1) < t)).mean())

    rowsB = [dict(thr=t, val=flag_b(val["text"], t), tune=flag_b(tune["text"], t), held=flag_b(held["text"], t))
             for t in THRESHOLDS]
    print("\nMethod B: extra 'Other' class (tuning half used for training, so 'tuning OOS' is optimistic)")
    print(f"{'threshold':>9} {'val flagged':>12} {'tuning OOS':>11} {'held-out OOS':>13}")
    for r in rowsB:
        print(f"{r['thr']:>9.2f} {r['val']:>12.0%} {r['tune']:>11.0%} {r['held']:>13.0%}")
    bestB = pick(rowsB)
    if bestB:
        print(f"\nMethod B best: threshold {bestB['thr']:.2f}  held-out OOS {bestB['held']:.0%}  val flagged {bestB['val']:.0%}")

    config = {"method": "max_probability_below_threshold", "threshold": float(best["thr"]), "seed": SEED,
              "tuned_on": "2026-10-28",
              "held_out_oos_flagged": best["held"], "validation_flagged": best["val"]}
    (ROOT / "model" / "config.json").write_text(json.dumps(config, indent=2))
    print("\nSaved model/config.json:", config)


if __name__ == "__main__":
    main()
