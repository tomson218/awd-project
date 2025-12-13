from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import seaborn as sns
import os
import time
import json
from sklearn.model_selection import train_test_split

# === 1. WCZYTAJ PRZYGOTOWANE DANE BINARNE ===
X_scaled = pd.read_csv("data/X_scaled.csv")           # już po MinMaxScaler
y_binary = pd.read_csv("data/y_binary.csv")

# Wybieramy naszą klasę: 1 = gospodarz nie przegra (H/D), 0 = goście wygrali
y = y_binary["HomeNoLose"].values

print(f"X shape: {X_scaled.shape}")
print(f"Rozkład klas (HomeNoLose):")
print(pd.Series(y).value_counts().sort_index())
print(f"Procent meczów gdzie gospodarz nie przegrał: {y.mean():.2%}")

X = X_scaled.values

# --- NOWY PODZIAŁ NA TRZY ZBIORY: TRAIN (80%), VALIDATION (10%), TEST (10%) ---

# Krok 1: Podział na 90% (tymczasowy) i 10% (testowy)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# Krok 2: Podział tymczasowego zbioru (90%) na 80% (treningowy) i 10% (walidacyjny)
# (1/9 z 90% to 10%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1/9, stratify=y_temp, random_state=42
)

# === ZAPIS PODZIELONYCH ZBIORÓW DO CSV ===
print("\nZapisywanie podzielonych danych do folderu data/split/...")
os.makedirs("data/split", exist_ok=True)

feature_names = X_scaled.columns.tolist()

# Tworzenie DataFrames dla Walidacji
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_val_df   = pd.DataFrame(X_val,   columns=feature_names)
X_test_df  = pd.DataFrame(X_test,  columns=feature_names)
y_train_df = pd.DataFrame(y_train, columns=["HomeNoLose"])
y_val_df   = pd.DataFrame(y_val,   columns=["HomeNoLose"])
y_test_df  = pd.DataFrame(y_test,  columns=["HomeNoLose"])

# Zapis do CSV
X_train_df.to_csv("data/split/X_train.csv", index=False)
X_val_df.to_csv(  "data/split/X_val.csv",   index=False)
X_test_df.to_csv( "data/split/X_test.csv",  index=False)
y_train_df.to_csv("data/split/y_train.csv", index=False)
y_val_df.to_csv(  "data/split/y_val.csv",   index=False)
y_test_df.to_csv( "data/split/y_test.csv",  index=False)

print("   → X_train.csv  :", X_train_df.shape)
print("   → X_val.csv    :", X_val_df.shape)
print("   → X_test.csv   :", X_test_df.shape)
print("   → y_train.csv  :", y_train_df.shape)
print("   → y_val.csv    :", y_val_df.shape)
print("   → y_test.csv   :", y_test_df.shape)
print("Zbiory zapisane!\n")

train_pool = Pool(X_train, y_train)
val_pool   = Pool(X_val, y_val)
test_pool  = Pool(X_test, y_test)


# === 3. GRID SEARCH – zoptymalizowany pod klasyfikację binarną ===
param_grid = {
    'iterations': [300, 500, 800, 1200],
    'depth': [5, 6, 7, 8],
    'learning_rate': [0.03, 0.05, 0.08, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [128, 254],
    #'scale_pos_weight': [1.0, 1.5, 2.0]   # ważenie klasy mniejszościowej (AwayWin)
}

model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',           # kluczowa metryka w zakładach!
    random_seed=42,
    thread_count=-1,
    verbose=False,
    early_stopping_rounds=100
)

print("\nGrid Search w toku (może potrwać 2-10 min)...")
grid_result = model.grid_search(param_grid, train_pool, cv=5, plot=False, verbose=1)

# === 4. NAJLEPSZE PARAMETRY Z CV ===
best_params = grid_result["params"]
cv_auc_mean = max(grid_result["cv_results"]["test-AUC-mean"])
cv_auc_std  = grid_result["cv_results"]["test-AUC-std"][np.argmax(grid_result["cv_results"]["test-AUC-mean"])]

print(f"\n5-fold CV – AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

# === 5. FINALNY MODEL ===
final_model = CatBoostClassifier(**best_params,
                                 loss_function='Logloss',
                                 eval_metric='AUC',
                                 random_seed=42,
                                 thread_count=-1,
                                 verbose=200)

print("\nTrenowanie finalnego modelu...")
start_time = time.time()
final_model.fit(train_pool, eval_set=val_pool, plot=False)
training_time = time.time() - start_time
print(f"Czas uczenia: {training_time:.2f} s")

# === 6. ZAPIS MODELU I SCALERA ===
os.makedirs("models", exist_ok=True)
final_model.save_model("models/catboost_home_no_lose.cbm")
print("Model zapisany → models/catboost_home_no_lose.cbm")


# === 7. ZAPIS WYNIKÓW FINALNYCH DO JSON ===
os.makedirs("results", exist_ok=True)

final_results = {
    "cv_auc_mean": cv_auc_mean,
    "cv_auc_std": cv_auc_std,
    "training_time_sec": training_time,
    "best_params": best_params
}

with open("results/binary_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("Wyniki zapisane → results/binary_results.json")





"""
         Cecha  Ważność (%)
9      MaxHome    14.391368
11     MaxAway    10.295922
8      OddAway     9.472144
16   HandiSize     6.756662
10     MaxDraw     6.469593
19       C_LTH     5.787165
0        C_LTA     5.228240
20       C_VHD     4.621096
23       C_HTB     3.397281
7      OddDraw     3.378007
22       C_PHB     2.984806
5    Form3Away     2.978149
1      HomeElo     2.679316
18   HandiAway     2.427039
13     Under25     2.394047
2      AwayElo     2.321979
15  MaxUnder25     2.203340
6    Form5Away     2.186233
21       C_VAD     2.020846
12      Over25     1.853391
17   HandiHome     1.668873
14   MaxOver25     1.635140
3    Form3Home     1.547789
4    Form5Home     1.301575
"""