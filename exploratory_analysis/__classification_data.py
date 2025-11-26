# __classification_data.py – FINALNA WERSJA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# === 1. WCZYTAJ DANE ===
X_processed = pd.read_csv("data/X_processed.csv").values
y_onehot = pd.read_csv("data/y_encoded.csv").values

# KONWERSJA Z ONE-HOT NA LICZBY (0,1,2)
y_encoded = np.argmax(y_onehot, axis=1)

print(f"Kształt y po konwersji: {y_encoded.shape}")
print(f"Unikalne klasy: {np.unique(y_encoded)}")
print(f"Rozkład klas:\n{pd.Series(y_encoded).value_counts().sort_index()}")

# === 2. PODZIAŁ 90/10 ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42
)

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# === 3. GRID SEARCH (Twoja siatka – działa!) ===
param_grid = {
    'iterations': [1000, 1500, 200],
    'depth': [6, 8, 10],
    'learning_rate': [0.1, 0.3, 0.5],
    'class_weights': [None, [1.0, 1.8, 1.1]]
}

model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    thread_count=-1,
    verbose=False
)

print("Grid Search...")
grid_result = model.grid_search(param_grid, X=train_pool, cv=5, plot=True, verbose=1)

# === 4. ŚREDNIA I ODCHYLENIE (2025 fix!) ===
cv_results = grid_result["cv_results"]
best_idx = np.argmax(cv_results["test-Accuracy-mean"])
cv_mean = cv_results["test-Accuracy-mean"][best_idx]
cv_std = cv_results["test-Accuracy-std"][best_idx]

print(f"\n5-fold CV – Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

# === 5. FINALNY MODEL ===
best_params = grid_result["params"]
final_model = CatBoostClassifier(**best_params, loss_function="MultiClass", random_seed=42)
final_model.fit(train_pool, verbose=200)

# === 6. ZAPIS MODELU ===
os.makedirs("models", exist_ok=True)
final_model.save_model("models/catboost_final.cbm")
print("Model zapisany!")

# === 7. TESTOWANIE ===
print("\nTestowanie na zbiorze testowym...")
y_test_pred = final_model.predict(X_test).flatten()

# y_test to liczby – ale na wszelki wypadek:
y_test_flat = y_test  # już jest liczbą

test_accuracy = accuracy_score(y_test_flat, y_test_pred)
test_f1 = f1_score(y_test_flat, y_test_pred, average='weighted')

print(f"Accuracy (test): {test_accuracy:.4f}")
print(f"F1-score: {test_f1:.4f}")

# === 8. MACIERZ I RAPORT ===
print("\nClassification Report:")
print(classification_report(y_test_flat, y_test_pred,
                          target_names=['Wygrana gospodarza (H)', 'Remis (D)', 'Wygrana gościa (A)']))

plt.figure(figsize=(7, 5.5))
sns.heatmap(confusion_matrix(y_test_flat, y_test_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'])
plt.ylabel('Rzeczywiste')
plt.xlabel('Przewidywane')
plt.title(f'Macierz pomyłek – Accuracy: {test_accuracy:.4f}')
plt.tight_layout()
plt.show()

# === 9. WAŻNOŚĆ CECH ===
print("\nTop najważniejszych cech:")
feature_importance = final_model.get_feature_importance()
numeric_cols = pd.read_csv("data/X_processed.csv").columns.tolist()

fi_df = pd.DataFrame({
    'Cecha': numeric_cols,
    'Ważność (%)': feature_importance
}).sort_values(by='Ważność (%)', ascending=False)

print(fi_df.head(10))

plt.figure(figsize=(10, 6))
top10 = fi_df.head(10)
sns.barplot(data=top10, x='Ważność (%)', y='Cecha', palette='viridis')
plt.title("Najważniejsze cechy")
plt.tight_layout()
plt.show()

# === 10. ZAPIS WYNIKÓW ===
results = {
    "cv_accuracy": float(cv_mean),
    "cv_std": float(cv_std),
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
    "best_params": best_params
}

import json   # ← DODAJ TO!

os.makedirs("results", exist_ok=True)
with open("results/final_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nWyniki zapisane: results/final_results.json")



# === 10 PRZYKŁADOWYCH MECZÓW – JAK CHCIAŁEŚ ===
print("\n" + "="*120)
print("10 PRZYKŁADOWYCH MECZÓW – RZECZYWISTE vs PRZEWIDYWANE")
print("="*120)

# 1. Weź 10 losowych rekordów z testowego
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=30, replace=False)

# 2. Przywróć nazwy kolumn (z oryginalnego X_processed.csv)
feature_names = pd.read_csv("data/X_processed.csv").columns.tolist()

# 3. DataFrame z wybranymi cechami
X_sample_df = pd.DataFrame(X_test[sample_idx], columns=feature_names)

# 4. Wybierz tylko kolumny, które chcesz pokazać
cols_to_show = ['HomeElo', 'AwayElo', 'HomeTarget', 'AwayTarget', 'C_LTA', 'C_HTB']
display_df = X_sample_df[cols_to_show]

# 5. Dodaj rzeczywiste i przewidywane wyniki
y_real = y_test[sample_idx]
y_pred = y_test_pred[sample_idx]

label_map = {0: 'Wygrana gościa', 1: 'Remis (D)', 2: 'Wygrana gospodarza (H)'}
display_df['Rzeczywista'] = [label_map[val] for val in y_real]
display_df['Przewidywana'] = [label_map[val] for val in y_pred]

# 6. Werdykt
display_df['WERDYKT'] = ['✓ POPRAWNIE' if r == p else '✗ BŁĄD'
                        for r, p in zip(y_real, y_pred)]

# 7. Reset indeksu + numeracja
display_df = display_df.reset_index(drop=True)
display_df.index = display_df.index + 1
display_df = display_df.rename_axis('Nr')

# 8. Pokaż tabelkę
print(display_df.to_string())