"""Loads the trained model ONCE and turns text into {category, uncertain, confidence}."""
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent.parent


class Classifier:
    def __init__(self, model_path: Path = ROOT / "model" / "model.joblib",
                 config_path: Path = ROOT / "model" / "config.json") -> None:
        self.model = joblib.load(model_path)
        self.config = json.loads(Path(config_path).read_text())
        self.threshold = float(self.config["threshold"])

    def predict(self, text: str) -> dict:
        probs = self.model.predict_proba([text])[0]
        best = int(probs.argmax())
        confidence = float(probs[best])
        return {
            "category": str(self.model.classes_[best]),   # best guess among the 4 labels
            "uncertain": bool(confidence < self.threshold),  # from the model's own probability
            "confidence": round(confidence, 3),
        }
