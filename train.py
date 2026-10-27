"""Train the Smart Notes classifier.

One command: python train.py

1. Reads data/train.csv (fit) and data/validation.csv (evaluate only).
2. Evaluates two baselines on validation: majority class and hand-written keyword rules.
3. Fits the ORIGINAL model (word TF-IDF + LogReg) and the FINAL model (word + character
   TF-IDF, stop words removed, + LogReg) on the training file ONLY, and reports both.
   The final model was chosen by grouped cross-validation on training data (cv_compare.py),
   not by looking at validation scores.
4. Prints accuracy / macro-F1, per-category results and a confusion matrix.
5. Saves the FINAL model to model/model.joblib (the app loads this file).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline, make_union

ROOT = Path(__file__).parent
LABELS = ["Study", "Money", "Travel", "Shopping"]
SEED = 42
MODEL_PATH = ROOT / "model" / "model.joblib"


def build_original_model() -> Pipeline:
    """First model (Phase 1): word 1-2-gram TF-IDF -> logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=10, max_iter=1000, random_state=SEED)),
    ])


def build_model() -> Pipeline:
    """Final model, chosen by grouped cross-validation on train.csv (see cv_compare.py).

    Two views of each note are joined: word 1-2 grams with English stop words removed,
    and character 3-5 grams (so 'coursework' and 'course' share pieces).
    """
    features = make_union(
        TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
    )
    return Pipeline([
        ("tfidf", features),
        ("clf", LogisticRegression(C=10, max_iter=1000, random_state=SEED)),
    ])


# ---- Baselines (comparison only, never inside the app) ---------------------------
KEYWORDS = {  # written from the category definitions in CATEGORY_GUIDE.md
    "Study": ["exam", "study", "course", "assignment", "tutorial", "professor", "homework", "revise", "lecture"],
    "Money": ["bill", "pay", "payment", "bank", "savings", "budget", "insurance", "invest", "transfer", "rent"],
    "Travel": ["flight", "train", "bus", "hotel", "trip", "itinerary", "ticket", "journey", "airport"],
    "Shopping": ["buy", "order", "purchase", "return", "cart", "shop", "store", "compare"],
}


def keyword_rule_predict(text: str, fallback: str) -> str:
    words = text.lower()
    scores = {c: sum(k in words for k in kws) for c, kws in KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return fallback
    return next(c for c in LABELS if scores[c] == best)  # ties -> first in LABELS order


def report(name: str, y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    print(f"{name:<38} accuracy={acc:.3f} macro-F1={f1:.3f}")


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    val = pd.read_csv(ROOT / "data" / "validation.csv")
    print(f"Training rows: {len(train)}  Validation rows: {len(val)}")
    print(f"sklearn {sklearn.__version__}, numpy {np.__version__}, seed {SEED}\n")

    majority = train["category"].mode()[0]
    report(f"Baseline: majority ('{majority}')", val["category"], [majority] * len(val))
    rule_pred = [keyword_rule_predict(t, majority) for t in val["text"]]
    report("Baseline: keyword rules", val["category"], rule_pred)

    original = build_original_model().fit(train["text"], train["category"])
    report("Trained v1: word TF-IDF + LogReg", val["category"], original.predict(val["text"]))

    model = build_model().fit(train["text"], train["category"])
    pred = model.predict(val["text"])
    report("Trained FINAL: word+char + LogReg", val["category"], pred)

    print("\nPer-category results, FINAL model (validation):")
    print(classification_report(val["category"], pred, labels=LABELS, digits=3, zero_division=0))
    print("Confusion matrix, FINAL model (rows = true, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(val["category"], pred, labels=LABELS), index=LABELS, columns=LABELS))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
