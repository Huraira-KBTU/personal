import argparse
import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a Telegram message is useful or trash."
    )
    parser.add_argument("text", help="Message text for classification")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file classifier.joblib not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba([args.text])[0]
    classes = model.classes_

    best_index = int(probabilities.argmax())
    result = {
        "label": classes[best_index],
        "probability": round(float(probabilities[best_index]), 4),
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
