import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOAD DATASET
DATASET_PATH = "../data/dataset_kredit.xlsx"

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded:")
print(df.head())

# FEATURES & TARGET
FEATURES = [
    "gaji_bulanan",
    "pengeluaran_bulanan",
    "jumlah_tanggungan",
    "jumlah_pinjaman",
    "tenor_bulan",
    "bunga_tahunan",
    "rasio_pengeluaran",
    "rasio_pinjaman",
    "rasio_cicilan",
    "rasio_tanggungan"
]

TARGET = "status_kredit"

X = df[FEATURES]
y = df[TARGET]

# LABEL ENCODING
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# TRAINING 8x RANDOM SPLIT
accuracies = []

print("\n=== TRAINING 8x RANDOM SPLIT ===")

for i in range(8):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42 + i,
        stratify=y_encoded
    )

    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        min_samples_split=30,
        random_state=42 + i
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Run {i+1} Accuracy: {acc:.4f}")

# VISUALISASI AKURASI
plt.figure()
plt.plot(range(1, 9), accuracies, marker='o')
plt.xticks(range(1, 9))
plt.xlabel("Training Run")
plt.ylabel("Accuracy")
plt.title("Accuracy per Training Run (8x Random Split)")
plt.grid(True)
plt.show()

# SUMMARY
print("\n=== SUMMARY ===")
print(f"Average Accuracy : {np.mean(accuracies):.4f}")
print(f"Std Deviation    : {np.std(accuracies):.4f}")

# FINAL EVALUATION (LAST RUN)
print("\n=== FINAL MODEL EVALUATION ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# SAVE MODEL & ENCODER
MODEL_PATH = "../model/model_kredit.pkl"
ENCODER_PATH = "../model/label_encoder.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print("\nModel & label encoder saved successfully.")
