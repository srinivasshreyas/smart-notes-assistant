"""Error analysis: WHY does the model get validation notes wrong? (read-only, fits nothing)

One command: python analyze_errors.py
For every mistake of the final model (and of the v1 model, for comparison) it prints the
note, true vs predicted label, the probabilities, and the 3 most similar TRAINING notes
(cosine similarity of the model's own TF-IDF features) as evidence.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from train import build_original_model

ROOT = Path(__file__).parent


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    val = pd.read_csv(ROOT / "data" / "validation.csv")
    model = joblib.load(ROOT / "model" / "model.joblib")
    old = build_original_model().fit(train["text"], train["category"])
    feats = model.named_steps["tfidf"]
    Xtr = feats.transform(train["text"])
    classes = list(model.classes_)

    pred = model.predict(val["text"])
    old_pred = old.predict(val["text"])
    proba = model.predict_proba(val["text"])
    wrong = [i for i in range(len(val)) if pred[i] != val["category"][i]]
    print(f"FINAL model: {len(wrong)} mistakes out of {len(val)}\n")
    for i in wrong:
        t = val["text"][i]
        sims = cosine_similarity(feats.transform([t]), Xtr)[0]
        top = sims.argsort()[::-1][:3]
        print(f"- \"{t}\"\n  true={val['category'][i]}  predicted={pred[i]}  "
              f"(v1 said {old_pred[i]})  top prob={proba[i].max():.2f}")
        print("  probs: " + "  ".join(f"{c}={p:.2f}" for c, p in zip(classes, proba[i])))
        for j in top:
            print(f"    similar train note ({sims[j]:.2f}, {train['category'][j]}): {train['text'][j]}")
        print()

    print("Mistakes of v1 that the final model fixed:")
    for i in range(len(val)):
        if old_pred[i] != val["category"][i] and pred[i] == val["category"][i]:
            print(f"  {val['text'][i]}   [{val['category'][i]}; v1 said {old_pred[i]}]")

    print("\nWords in validation notes that never occur in train.csv:")
    seen = set(" ".join(train["text"].str.lower()).replace(".", " ").split())
    unseen = sorted({w for t in val["text"] for w in t.lower().replace(".", " ").replace(",", " ").split() if w not in seen})
    print("  " + ", ".join(unseen))


if __name__ == "__main__":
    main()
