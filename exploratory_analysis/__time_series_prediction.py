"""import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

X = pd.read_csv('data/X_processed.csv').values
y = pd.read_csv('data/y_encoded.csv').values

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Całkowita liczba próbek
n_samples = len(X)

# Oblicz indeks podziału: 90% na trening, 10% na test
train_end = int(0.90 * n_samples)

# === PODZIAŁ BEZ MIESZANIA ===
X_train = X[:train_end]
y_train = y[:train_end]

X_test  = X[train_end:]
y_test  = y[train_end:]

# === SPRAWDŹ ===
print(f"Całość: {n_samples}")
print(f"Train:  {len(X_train)} ({len(X_train)/n_samples:.1%}) → indeksy [0 : {train_end})")
print(f"Test:   {len(X_test)}  ({len(X_test)/n_samples:.1%}) → indeksy [{train_end} : {n_samples})")

# Sprawdzenie poprawności podziału
assert len(X_train) + len(X_test) == n_samples, "Błąd w podziale danych!"
print("Podział poprawny!")
assert len(X_train) + + len(X_test) == n_samples

# Utwórz folder
os.makedirs("data/splits", exist_ok=True)

# === OPCJA 1: ZAPIS DO CSV (czytelne, uniwersalne) ===
print("\nZAPISUJĘ DO CSV...")

# Konwersja na DataFrame (żeby CSV miał nagłówki)
X_train_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(X_train.shape[1])])
X_test_df  = pd.DataFrame(X_test,  columns=[f"feat_{i}" for i in range(X_test.shape[1])])

# y – nazwy klas
y_columns = ['FTResult_H', 'FTResult_D', 'FTResult_A']
y_train_df = pd.DataFrame(y_train, columns=y_columns)
y_test_df  = pd.DataFrame(y_test,  columns=y_columns)

# ZAPIS
X_train_df.to_csv("data/splits/X_train.csv", index=False)
X_test_df.to_csv("data/splits/X_test.csv", index=False)

y_train_df.to_csv("data/splits/y_train.csv", index=False)
y_test_df.to_csv("data/splits/y_test.csv", index=False)

print("Zapisano do CSV!")"""