# __classification_binary.py – FINALNA WERSJA BINARNA (HomeNoLose vs AwayWin)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import json
from sklearn.model_selection import train_test_split

# 1. WCZYTANIE MODELU
final_model = CatBoostClassifier()                    # pusty obiekt
final_model.load_model("models/catboost_home_no_lose.cbm")   # ← tu podajesz swoją ścieżkę

# Load the split you just saved
#X_train = pd.read_csv("data/split/X_train.csv")
X_test  = pd.read_csv("data/split/X_test.csv")
#y_train = pd.read_csv("data/split/y_train.csv")["HomeNoLose"]   # as Series
y_test  = pd.read_csv("data/split/y_test.csv") ["HomeNoLose"]   # as Series

X_scaled = pd.read_csv("data/X_scaled.csv")

with open("results/binary_results.json", "r") as f:
    binary_results = json.load(f)

cv_auc_mean = binary_results["cv_auc_mean"]
training_time = binary_results["training_time_sec"]
best_params = binary_results["best_params"]
cv_auc_std = binary_results["cv_auc_std"]

print(cv_auc_mean)
print(training_time)
print(best_params)



# === 7. PREDYKCJE I METRYKI ===
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)

print(f"\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
print(f"Accuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# === 8. MACIERZ POMYŁEK ===
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Goście wygrali (A)', 'Gospodarz nie przegrał (H/D)'],
            yticklabels=['Goście wygrali (A)', 'Gospodarz nie przegrał (H/D)'])
plt.ylabel('Rzeczywiste')
plt.xlabel('Przewidywane')
plt.title(f'Macierz pomyłek – AUC: {auc:.4f}')
plt.tight_layout()
plt.show()

# === 9. WAŻNOŚĆ CECH ===
feature_names = X_scaled.columns.tolist()
fi = final_model.get_feature_importance(type='PredictionValuesChange')
fi_df = pd.DataFrame({'Cecha': feature_names, 'Ważność (%)': fi}).sort_values(by='Ważność (%)', ascending=False)

print("\nTop 10 najważniejszych cech:")
print(fi_df.head(30))

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df.head(10), x='Ważność (%)', y='Cecha', palette='viridis')
plt.title("Najważniejsze cechy – CatBoost")
plt.tight_layout()
plt.show()

# === 10. ZAPIS WYNIKÓW ===
results = {
    "task": "HomeNoLose vs AwayWin",
    "cv_auc_mean": float(cv_auc_mean),
    "cv_auc_std": float(cv_auc_std),
    "test_accuracy": float(acc),
    "test_f1": float(f1),
    "test_auc": float(auc),
    "training_time_sec": float(training_time),
    "best_params": best_params
}

os.makedirs("results", exist_ok=True)
with open("results/binary_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nWyniki zapisane → results/testing_results.json")

# === 11. 15 PRZYKŁADOWYCH MECZÓW Z PROBABILISTYKAMI ===
print("\n" + "="*130)
print("15 LOSOWYCH MECZÓW Z TESTU – RZECZYWISTE vs PRZEWIDYWANE + PRAWDOPODOBIEŃSTWO")
print("="*130)

np.random.seed(42)
idx = np.random.choice(len(X_test), size=15, replace=False)

display_df = X_scaled.iloc[idx].copy()
display_df = display_df[['C_LTA', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'OddDraw', 'OddAway', 'MaxHome', 'MaxDraw', 'MaxAway', 'Over25',
                'Under25', 'MaxOver25', 'MaxUnder25', 'HandiSize', 'HandiHome', 'HandiAway', 'C_LTH', 'C_VHD', 'C_VAD', 'C_PHB',

                                                                                                                                               'C_HTB']]

display_df['P(HomeNoLose)'] = y_pred_proba[idx].round(4)
display_df['Przewidywana'] = ['HomeNoLose' if p >= 0.5 else 'AwayWin' for p in y_pred_proba[idx]]
display_df['Rzeczywista'] = ['HomeNoLose' if real == 1 else 'AwayWin' for real in y_test[idx]]
display_df['WERDYKT'] = ['POPRAWNIE' if pred == real else 'BŁĄD'
                         for pred, real in zip(display_df['Przewidywana'], display_df['Rzeczywista'])]

display_df = display_df.reset_index(drop=True)
display_df.index += 1
display_df = display_df.rename_axis('Nr')

print(display_df.to_string(float_format="{:.4f}".format))