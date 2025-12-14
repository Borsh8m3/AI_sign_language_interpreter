import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import joblib, json

from utils import row_to_features, FEAT_DIM

CSV_PATH = "gestures.csv"
CLEAN_CSV_PATH = "gestures_clean.csv"
MODEL_PATH = "model.joblib"
LABELS_PATH = "labels.json"

EXPECTED_LEN = 1 + FEAT_DIM 

def clean_csv(in_path, out_path):
    """Zapisuje tylko poprawne wiersze o długości 64 kolumn."""
    good, bad = 0, 0
    with open(in_path, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        for row in r:
            if len(row) == EXPECTED_LEN:
                w.writerow(row)
                good += 1
            else:
                bad += 1
    print(f"dobre: {good}, złe: {bad} -> zapisano {out_path}")

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    y = df.iloc[:, 0].astype(str).values
    X = np.vstack([row_to_features(df.iloc[i, :].tolist()) for i in range(len(df))])
    return X, y

# 1) Najpierw czyścimy plik (to naprawia błąd Expected 64 fields...)
clean_csv(CSV_PATH, CLEAN_CSV_PATH)

# 2) Wczytujemy już czyste dane
X, y = load_dataset(CLEAN_CSV_PATH)

# 3) Pokaż liczbę próbek na klasę (super do debugowania)
print("\nLiczba próbek na literę:")
print(pd.Series(y).value_counts())

# 4) Split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5) Model
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", probability=True)
)
clf.fit(Xtr, ytr)

# 6) Ewaluacja
pred = clf.predict(Xte)
print("\nAccuracy:", accuracy_score(yte, pred))
print(classification_report(yte, pred))

# 7) Zapis
joblib.dump(clf, MODEL_PATH)
labels = sorted(list(set(y)))
with open(LABELS_PATH, "w") as f:
    json.dump({"labels": labels}, f)

print("\nZapisano:", MODEL_PATH, LABELS_PATH)
