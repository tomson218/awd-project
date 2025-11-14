from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt




# --- PO PRZETWARZANIU (masz X_processed, y_encoded) ---

# Konwersja y_encoded (one-hot) na etykiety numeryczne do stratify
y_labels = np.argmax(y_encoded, axis=1)  # 0=H, 1=D, 2=A
label_map = {0: 'H', 1: 'D', 2: 'A'}

# Podział na train (90%) i test (10%) z zachowaniem proporcji klas
X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X_processed, y_encoded, test_size=0.1, random_state=42, stratify=y_labels
)

# Konwersja etykiet train do liczb dla StratifiedKFold
y_train_labels = np.argmax(y_train_enc, axis=1)

print(f"\nPodział danych:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Rozkład klas w train: {np.bincount(y_train_labels)}")
print(f"Rozkład klas w test:  {np.bincount(np.argmax(y_test_enc, axis=1))}")

# Przykład modelu
model = LogisticRegression(max_iter=1000, multi_class='multinomial')

# 5-fold CV z zachowaniem proporcji
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

print("\n=== 5-Fold Cross Validation ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_labels), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train_enc[train_idx], y_train_enc[val_idx]

    # Uczenie
    model.fit(X_fold_train, y_fold_train)

    # Predykcja
    y_pred = model.predict(X_fold_val)
    y_val_labels = np.argmax(y_fold_val, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_val_labels, y_pred_labels)
    accuracies.append(acc)

    print(f"Fold {fold}: Accuracy = {acc:.4f}")

# Wyniki do sprawozdania
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\nŚrednia dokładność (5-fold CV): {mean_acc:.4f} ± {std_acc:.4f}")

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000]
}

grid = GridSearchCV(
    LogisticRegression(multi_class='multinomial', max_iter=1000),
    param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train_enc)
print(f"Najlepsze parametry: {grid.best_params_}")
print(f"Najlepsza dokładność CV: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train_enc,
    cv=skf, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Trening')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='green', label='Walidacja')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')

plt.title('Krzywe uczenia - Logistic Regression')
plt.xlabel('Rozmiar zbioru treningowego')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

best_model = grid.best_estimator_  # lub wybrany ręcznie
y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(np.argmax(y_test_enc, axis=1), np.argmax(y_test_pred, axis=1))
print(f"\nDokładność na zbiorze testowym: {test_acc:.4f}")