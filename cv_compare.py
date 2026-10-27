"""Choose a model fix using ONLY train.csv (grouped cross-validation).

Why grouped? train.csv is built from templates ("Confirm the bus seat for <city>").
If two notes from the same template land in different folds, the model has
effectively seen the test note. Grouping keeps every template in ONE fold, so each
test fold contains wording the model has never seen -- the same situation as real
life and as validation.csv. validation.csv is NOT read by this script.

One command: python cv_compare.py
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline, make_union

ROOT = Path(__file__).parent
SEED = 42

DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight"
CITIES = r"kochi|delhi|mumbai|pune|chennai|bengaluru|jaipur"
SUBJECTS = r"biology|chemistry|mathematics|statistics|economics|history|python|machine learning"
ITEMS = (r"backpack|keyboard|headphones|water bottle|desk lamp|laptop stand|notebook|"
         r"shoes|clothes|gift")


def template_of(text: str) -> str:
    """Mask the parts that vary inside one template, so the template is the group."""
    t = text.lower()
    t = re.sub(r"₹[\d,]+", "<amt>", t)
    t = re.sub(r"\bthe \d+(st|nd|rd|th)\b", "<day>", t)
    t = re.sub(rf"\b({DAYS})\b", "<day>", t)
    t = re.sub(r"\b\d+ (am|pm)\b", "<time>", t)
    t = re.sub(rf"\b({CITIES})\b", "<city>", t)
    t = re.sub(rf"\b({SUBJECTS})\b", "<subj>", t)
    t = re.sub(rf"\b({ITEMS})\b", "<item>", t)
    return t


def make(name: str) -> Pipeline:
    word = dict(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    if name == "A. current (word 1-2gram, C=10)":
        v, c = TfidfVectorizer(**word), LogisticRegression(C=10, max_iter=1000, random_state=SEED)
    elif name == "B. lower C=1":
        v, c = TfidfVectorizer(**word), LogisticRegression(C=1, max_iter=1000, random_state=SEED)
    elif name == "C. class_weight=balanced":
        v, c = TfidfVectorizer(**word), LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=SEED)
    elif name == "D. drop English stop words":
        v, c = TfidfVectorizer(stop_words="english", **word), LogisticRegression(C=10, max_iter=1000, random_state=SEED)
    elif name == "E. char n-grams (3-5)":
        v, c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), LogisticRegression(C=10, max_iter=1000, random_state=SEED)
    elif name == "F. stop words + char n-grams":
        v = make_union(TfidfVectorizer(stop_words="english", **word),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
        c = LogisticRegression(C=10, max_iter=1000, random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([("tfidf", v), ("clf", c)])


CANDIDATES = [
    "A. current (word 1-2gram, C=10)", "B. lower C=1", "C. class_weight=balanced",
    "D. drop English stop words", "E. char n-grams (3-5)", "F. stop words + char n-grams",
]


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    groups = train["text"].map(template_of)
    gid = groups.astype("category").cat.codes.to_numpy()
    print(f"{len(train)} training notes fall into {groups.nunique()} templates "
          f"(largest template: {groups.value_counts().max()} notes)")
    k = min(5, groups.nunique())
    cv = GroupKFold(n_splits=k)
    print(f"GroupKFold, {k} folds, training data only\n")
    print(f"{'candidate':<36}{'cv accuracy':>12}{'cv macro-F1':>13}")
    rows = []
    for name in CANDIDATES:
        accs, f1s = [], []
        for tr, te in cv.split(train["text"], train["category"], gid):
            m = make(name).fit(train["text"].iloc[tr], train["category"].iloc[tr])
            p = m.predict(train["text"].iloc[te])
            y = train["category"].iloc[te]
            accs.append((p == y).mean()); f1s.append(f1_score(y, p, average="macro"))
        rows.append((name, np.mean(accs), np.mean(f1s)))
        print(f"{name:<36}{np.mean(accs):>12.3f}{np.mean(f1s):>13.3f}")
    best = max(rows, key=lambda r: r[2])
    print(f"\nBest by grouped-CV macro-F1: {best[0]}")


if __name__ == "__main__":
    main()
