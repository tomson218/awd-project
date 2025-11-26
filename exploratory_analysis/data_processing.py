from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import os

# Wczytaj dane
df = pd.read_csv('../csv/Matches_cleaned.csv')

# --- NUMERYCZNE (do MinMax [0,1]) ---
#poprawic to
"""numeric_cols = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'HomeShots', 'AwayShots',
                'HomeTarget', 'AwayTarget', 'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners',
                'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed']"""

#numeric_cols = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
#numeric_cols = ['HomeElo', 'AwayElo', 'HomeTarget', 'AwayTarget', 'HomeShots', 'AwayShots',]

"""numeric_cols = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'HomeShots', 'AwayShots',
                'HomeTarget', 'AwayTarget', 'HomeFouls', 'AwayFouls', 'HomeCorners', 'AwayCorners',
                'HomeYellow', 'AwayYellow', 'HomeRed', 'AwayRed', 'OddHome', 'OddDraw', 'OddAway', 'MaxHome', 'MaxDraw',
                'MaxAway', 'Over25', 'Under25', 'MaxOver25', 'MaxUnder25', 'HandiSize', 'HandiHome', 'HandiAway',
                'C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']"""

"""numeric_cols = ['HomeTarget', 'AwayTarget', 'HomeCorners', 'AwayCorners',
                'C_LTA', 'C_PHB', 'HomeElo', 'AwayElo', 'C_HTB']"""

numeric_cols = ['HomeTarget', 'AwayTarget',
                'C_LTA',  'HomeElo', 'AwayElo', 'C_HTB']


# --- CECHY (X) ---
X = df[numeric_cols].copy()

# --- ETYKIETY (y) ---
target_col = 'FTResult'  # H, D, A
y = df[target_col].copy()

# 1. Skalowanie X → [0,1]
scaler = MinMaxScaler()
X_processed = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)

# --- ENKODER DLA y (OneHot) ---
y_encoder = OneHotEncoder(sparse_output=False, categories=[['H', 'D', 'A']])


def display_data_before_preprocessing():
    print("=== DANE PRZED PRZETWORZENIEM ===")
    print(f"X shape (przed): {X.shape}")
    print(f"y unikalne: {sorted(y.unique())}")
    print(f"y rozkład:\n{pd.Series(y).value_counts().sort_index()}")

    print("\n--- X (oryginalne wartości numeryczne) ---")
    print(X.head())

    print("\n--- y (oryginalne etykiety) ---")
    print(y.head())


def display_data_after_preprocessing(X_processed, y_encoded):
    # X po skalowaniu
    X_df = pd.DataFrame(X_processed, columns=numeric_cols)
    print("\n=== X po MinMaxScaler (wartości w [0, 1]) ===")
    print(X_df.head())
    print(f"Min wartości: {X_df.min().min():.4f}, Max wartości: {X_df.max().max():.4f}")

    # y po one-hot
    y_df = pd.DataFrame(y_encoded, columns=y_encoder.categories_[0])
    print("\n=== y po OneHotEncoder ===")
    print(y_df.head())
    print(f"Suma po kolumnach (powinna być 1.0): {y_df.sum(axis=1).unique()}")



def encode_labels(y, y_encoder):
    #Koduje etykiety klas na one-hot
    y_encoded = y_encoder.fit_transform(y.values.reshape(-1, 1))
    return y_encoded, y_encoder


# === WYKONANIE ===
if __name__ == "__main__":
    # 1. Przed przetworzeniem
    display_data_before_preprocessing()

    # 3. Przetwarzanie y
    y_encoded, y_encoder = encode_labels(y, y_encoder)

    # 4. Po przetworzeniu
    display_data_after_preprocessing(X_processed, y_encoded)

    # === ZAPIS DO CSV + .pkl ===
    os.makedirs("data", exist_ok=True)

    # ZAPIS X
    pd.DataFrame(X_processed, columns=numeric_cols).to_csv(
        "data/X_processed.csv", index=False
    )
    print("Zapisano: data/X_processed.csv")

    # === NAZWY KOLUMN DLA y PO ONEHOT ===
    y_columns = ['FTResult_H', 'FTResult_D', 'FTResult_A']

    # ZAPIS y
    pd.DataFrame(y_encoded, columns=y_columns).to_csv(
        "data/y_encoded.csv", index=False
    )
    print("Zapisano: data/y_encoded.csv")

    # Zwracamy gotowe dane (opcjonalnie)
    print("\nPrzetwarzanie zakończone. Dane gotowe do modelu.")
