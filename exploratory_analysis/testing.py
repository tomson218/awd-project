from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, log_loss
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

final_model = CatBoostClassifier()
final_model.load_model("models/catboost_home_no_lose.cbm")


X_test  = pd.read_csv("data/split/X_test.csv")
y_test  = pd.read_csv("data/split/y_test.csv") ["HomeNoLose"]

X_scaled = pd.read_csv("data/X_scaled.csv")

with open("results/binary_results.json", "r") as f:
    binary_results = json.load(f)

cv_auc_mean = binary_results["cv_auc_mean"]
training_time = binary_results["training_time_sec"]
best_params = binary_results["best_params"]
cv_auc_std = binary_results["cv_auc_std"]

val_auc = binary_results["final_validation_metrics"]["AUC"]
val_accuracy = binary_results["final_validation_metrics"]["Accuracy"]
val_precision = binary_results["final_validation_metrics"]["Precision"]
val_recall = binary_results["final_validation_metrics"]["Recall"]
val_logloss = binary_results["final_validation_metrics"]["Logloss"]


#print(cv_auc_mean)
#print(training_time)
print(best_params)


y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)
logloss_val = log_loss(y_test, y_pred_proba)

print(f"\nWYNIKI TESTOWANIA")
print(f"Accuracy : {acc:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print(f"LogLoss   : {logloss_val:.4f}")

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

results = {
    "task": "HomeNoLose vs AwayWin",
    "cv_auc_mean": float(cv_auc_mean),
    "cv_auc_std": float(cv_auc_std),
    "test_accuracy": float(acc),
    "test_f1": float(f1),
    "test_auc": float(auc),
    "training_time_sec": float(training_time),
    "best_params": best_params,
    "final_validation_metrics": {
        "AUC": float(val_auc),
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "Logloss": float(val_logloss)
    }
}

os.makedirs("results", exist_ok=True)
with open("results/binary_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


print("\n\n20 MECZÓW Z DANYCH TESTOWYCH")

np.random.seed(42)
idx = np.random.choice(len(X_test), size=20, replace=False)

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

print(display_df.to_string(float_format="{:.2f}".format))

os.makedirs("results", exist_ok=True)
csv_path = "results/15_przykladowych_meczow_test.csv"

display_df.to_csv(csv_path, index=True, encoding='utf-8-sig')