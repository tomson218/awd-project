from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import joblib

df = pd.read_csv('../csv/Exploratory_matches_cleaned.csv')

numeric_cols = ['C_LTA', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away', 'OddDraw', 'OddAway', 'OddHome', 'MaxHome', 'MaxDraw', 'MaxAway', 'Over25',
                'Under25', 'MaxOver25', 'MaxUnder25', 'HandiSize', 'HandiHome', 'HandiAway', 'C_LTH', 'C_VHD', 'C_VAD', 'C_PHB', 'C_HTB']

X = df[numeric_cols].copy()
y = df['FTResult'].copy()

y_binary = pd.DataFrame({
    "HomeNoLose": ((y == 'H') | (y == 'D')).astype(int),
    "AwayWin":    (y == 'A').astype(int)
}, index=df.index)

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=numeric_cols,
    index=X.index
)

os.makedirs("data", exist_ok=True)

X_scaled.to_csv("data/X_scaled.csv", index=False)

y_binary.to_csv("data/y_binary.csv", index=False)

joblib.dump(scaler, "data/scaler.pkl")

print("\nRozkład klas:")
print(y_binary.mean())