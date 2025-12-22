from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import joblib

# Wczytaj dane
df = pd.read_csv('../csv/Exploratory_matches_cleaned.csv')

# --- NUMERYCZNE CECHY ---
"""numeric_cols = ['C_LTA', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'OddDraw', 'OddAway', 'OddHome', 'MaxHome', 'MaxDraw', 'MaxAway', 'Over25',
                'Under25', 'MaxOver25', 'MaxUnder25', 'HandiSize', 'HandiHome', 'HandiAway', 'C_LTH', 'C_VHD', 'C_VAD', 'C_PHB', 'C_HTB']
                """

numeric_cols = ["OddHome", "MaxDraw", "MaxAway", "OddDraw", "Form5Away", "MaxHome", "MaxUnder25", "HandiSize",
                "Form5Home", "OddAway", "Under25", "C_LTH", "HomeElo", "C_VHD", "AwayElo", "C_LTA"]


X = df[numeric_cols].copy()
y = df['FTResult'].copy()

# --- TWOJE 2 KLASY BINARNE (to jest dokładnie to, czego chcesz) ---
y_binary = pd.DataFrame({
    "HomeNoLose": ((y == 'H') | (y == 'D')).astype(int),   # 1 = gospodarz nie przegra
    "AwayWin":    (y == 'A').astype(int)                  # 1 = przyjezdni wygrali
}, index=df.index)

# --- Skalowanie cech numerycznych do [0, 1] ---
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=numeric_cols,
    index=X.index
)

# --- Zapisanie wszystkiego ---
os.makedirs("data", exist_ok=True)

# 1. Skalowane cechy
X_scaled.to_csv("data/X_scaled.csv", index=False)

# 2. Etykiety binarne (to jest Twój y do modelu!)
y_binary.to_csv("data/y_binary.csv", index=False)

# 3. Scaler – bardzo ważny przy predykcjach na nowych danych!
joblib.dump(scaler, "data/scaler.pkl")

print("Gotowe! Pliki zapisane w folderze data/:")
print("   - X_scaled.csv      → cechy po MinMaxScaler")
print("   - y_binary.csv      → dwie klasy: HomeNoLose i AwayWin")
print("   - scaler.pkl        → do użycia przy predykcjach live")

# Przykładowy rozkład klas
print("\nRozkład klas:")
print(y_binary.mean())  # % meczów gdzie gospodarz nie przegrał / gdzie przyjezdni wygrali