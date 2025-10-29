import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import random

# Set the path to the file you'd like to load
MATCHES_FILE_PATH = "Matches.csv"
ELO_RATINGS_FILE_PATH = "EloRatings.csv"

# Load the latest version
def loading_file(file_path):
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "adamgbor/club-football-match-data-2000-2025",
        file_path,
    )


def generate_random_time():
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    return f"{hours:02d}:{minutes:02d}:00"


def filter_data(df):
    df = df.copy()  # nie modyfikujemy oryginalnego

    # 1. Konwersja MatchDate na datetime (jeśli jeszcze nie jest)
    if not pd.api.types.is_datetime64_any_dtype(df['MatchDate']):
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')

    # 2. Filtr lig: I1, SP1, D1, F1, N1
    top_leagues = ['I1', 'SP1', 'D1', 'F1', 'N1']
    mask_league = df['Division'].isin(top_leagues)

    # 3. Filtr dat: 01.08.2022 – 30.06.2025
    start_date = pd.Timestamp('2022-08-01')
    end_date = pd.Timestamp('2025-06-30')
    mask_date = df['MatchDate'].between(start_date, end_date, inclusive='both')

    # 4. Zastosuj filtry
    df_filtered = df[mask_league & mask_date].reset_index(drop=True)

    # 5. Opcjonalnie: usuń rekordy z brakującymi kluczowymi danymi (np. wynik meczu)
    df_filtered = df_filtered.dropna(subset=['FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway', 'HTResult', "HomeTeam", "AwayTeam"], how='any')

    print(f"Po filtrze: {len(df_filtered)} meczów (z {len(df)} oryginalnych)")
    print(f"Ligi: {sorted(df_filtered['Division'].unique())}")
    print(f"Zakres dat: {df_filtered['MatchDate'].min()} → {df_filtered['MatchDate'].max()}")

    return df_filtered


matches_df = loading_file(MATCHES_FILE_PATH)
elo_ratings_df = loading_file(ELO_RATINGS_FILE_PATH)

matches_filtered = filter_data(matches_df)


def clean_data(matches_filtered):
    if 'MatchTime' in matches_filtered.columns:
        # spróbuj skonwertować na czas, błędne → NaT
        matches_filtered['MatchTime'] = pd.to_datetime(
            matches_filtered['MatchTime'], format='%H:%M:%S', errors='coerce'
        ).dt.time

        # maska niepoprawnych wartości
        invalid_mask = matches_filtered['MatchTime'].isna()
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:
            #print(f"Kolumna 'MatchTime': {num_invalid} niepoprawnych wartości → zastąpiono losowym czasem")

            # funkcja generująca losowy czas
            def random_time():
                h = np.random.randint(0, 24)
                m = np.random.randint(0, 60)
                s = np.random.randint(0, 60)
                return datetime.time(h, m, s)

            # zastąp wszystkie błędne wartości losowym czasem
            matches_filtered.loc[invalid_mask, 'MatchTime'] = [random_time() for _ in range(num_invalid)]

    # --- 8️⃣ Kolumny Elo ---
    elo_cols = ["HomeElo", "AwayElo"]
    for col in elo_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()
            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} brakujących wartości → zastąpiono średnią = {mean_value:.2f}")
                matches_filtered.loc[invalid_mask, col] = mean_value

    # --- 6️⃣ Kolumny Form3 ---
    forms_3_cols = ["Form3Home", "Form3Away"]
    for col in forms_3_cols:
        if col in matches_filtered.columns:
            # konwersja na liczby całkowite
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')
            # maska błędnych danych: NaN lub poza zakresem 0-9
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0) | (matches_filtered[col] > 9)
            num_invalid = invalid_mask.sum()
            mean_value = int(round(matches_filtered[col].mean(skipna=True)))
            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} błędnych wartości → zastąpiono średnią = {mean_value}")
                matches_filtered.loc[invalid_mask, col] = mean_value
            # upewnij się, że typ int
            matches_filtered[col] = matches_filtered[col].astype(int)

    # --- 7️⃣ Kolumny Form5 ---
    forms_5_cols = ["Form5Home", "Form5Away"]
    for col in forms_5_cols:
        if col in matches_filtered.columns:
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0) | (matches_filtered[col] > 15)
            num_invalid = invalid_mask.sum()
            mean_value = int(round(matches_filtered[col].mean(skipna=True)))
            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} błędnych wartości → zastąpiono średnią = {mean_value}")
                matches_filtered.loc[invalid_mask, col] = mean_value
            matches_filtered[col] = matches_filtered[col].astype(int)

    # --- 5️⃣ Kolumny ze statystykami meczowymi ---
    match_stats_cols = [
        'HomeShots', 'AwayShots',
        'HomeTarget', 'AwayTarget',
        'HomeFouls', 'AwayFouls',
        'HomeCorners', 'AwayCorners',
        'HomeYellow', 'AwayYellow',
        'HomeRed', 'AwayRed'
    ]

    for col in match_stats_cols:
        if col in matches_filtered.columns:
            # zamień na liczby, wszystko co nie da się przekonwertować → NaN
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')

            # policz średnią i zaokrąglij do najbliższej liczby całkowitej
            mean_value = int(round(matches_filtered[col].mean(skipna=True)))

            # maska błędnych danych
            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} nieliczbowych lub pustych wartości → zastąpiono średnią = {mean_value}")
                matches_filtered.loc[invalid_mask, col] = mean_value

            # upewnij się, że kolumna ma typ int
            matches_filtered[col] = matches_filtered[col].astype(int)

    # --- 4️⃣ Kolumny z kursami i over/under ---
    odds_cols = [
        'OddHome', 'OddDraw', 'OddAway',
        'MaxHome', 'MaxDraw', 'MaxAway',
        'Over25', 'Under25',
        'MaxOver25', 'MaxUnder25'
    ]

    for col in odds_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 1.0)
            num_invalid = invalid_mask.sum()
            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} ujemnych lub pustych wartości → średnia = {mean_value:.4f}")
                matches_filtered.loc[invalid_mask, col] = mean_value

    handi_size_col = ["HandiSize"]

    for col in handi_size_col:
        if col in matches_filtered.columns:
            # Spróbuj przekonwertować wartości na liczbowe (nie dające się przekonwertować staną się NaN)
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')

            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                #print(
                    #f"Kolumna '{col}': {num_invalid} nieliczbowych lub pustych wartości → zastąpiono średnią = {mean_value:.4f}")
                matches_filtered.loc[invalid_mask, col] = mean_value

    # --- 2️⃣ Kolumny z handicapami ---
    handicap_cols = ["HandiHome", "HandiAway"]

    for col in handicap_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0)
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                #print(
                 #   f"Kolumna '{col}': {num_invalid} ujemnych lub pustych wartości → zastąpiono średnią = {mean_value:.4f}")
                matches_filtered.loc[invalid_mask, col] = mean_value

    # Lista kolumn, które mają być sprawdzone
    prob_cols = ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']

    for col in prob_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)

            # maska błędnych lub brakujących wartości
            invalid_mask = (matches_filtered[col].isna()) | (matches_filtered[col] < 0.0001) | (matches_filtered[col] > 0.9999)
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                #print(f"Kolumna '{col}': {num_invalid} błędnych wartości → zastąpiono średnią = {mean_value:.4f}")
                matches_filtered.loc[invalid_mask, col] = mean_value

    return matches_filtered


matches_before_clearing = matches_df.copy()

matches_after_cleaning = clean_data(matches_filtered)

# 3. Zapis do nowego pliku CSV
matches_after_cleaning.to_csv("Matches_cleaned.csv", index=False)

print("Dane zostały zapisane do Matches_cleaned.csv")

#print(matches_after_cleaning[["MaxOver25", "MaxUnder25"]])




