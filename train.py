from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "Crop_recommendation.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    if not DATA.exists():
        raise SystemExit(
            f"Dataset not found: {DATA}\n"
            "Download the Crop Recommendation CSV and place it at backend/data/Crop_recommendation.csv"
        )

    df = pd.read_csv(DATA)
    # Support common alternative target column names.
    if "label" not in df.columns:
        for candidate in ("crop", "Crop", "target"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "label"})
                break

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. CSV columns found: {list(df.columns)}")

    df = df[FEATURES + ["label"]].dropna()
    X, y = df[FEATURES], df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"Rows used: {len(df)}")
    print(f"Classes: {y.nunique()}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test macro F1: {macro_f1:.3f}")
    print("\nClassification report:\n", classification_report(y_test, preds))

    joblib.dump({"model": model, "features": FEATURES}, MODEL_DIR / "crop_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"test_accuracy": accuracy, "test_macro_f1": macro_f1,
                   "rows": int(len(df)), "classes": int(y.nunique())}, f, indent=2)
    print(f"\nSaved model to {MODEL_DIR / 'crop_model.joblib'}")

if __name__ == "__main__":
    main()
