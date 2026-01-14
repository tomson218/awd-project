import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
import random

MATCHES_FILE_PATH = "Matches.csv"
ELO_RATINGS_FILE_PATH = "EloRatings.csv"

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
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df['MatchDate']):
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')

    top_leagues = ['I1', 'SP1', 'D1', 'F1', 'N1']
    mask_league = df['Division'].isin(top_leagues)

    start_date = pd.Timestamp('2020-08-01')
    end_date = pd.Timestamp('2025-06-30')
    mask_date = df['MatchDate'].between(start_date, end_date, inclusive='both')

    df_filtered = df[mask_league & mask_date].reset_index(drop=True)

    df_filtered = df_filtered.dropna(subset=['FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway', 'HTResult', "HomeTeam", "AwayTeam"], how='any')



    return df_filtered


matches_df = loading_file(MATCHES_FILE_PATH)
elo_ratings_df = loading_file(ELO_RATINGS_FILE_PATH)

matches_filtered = filter_data(matches_df)


def clean_data(matches_filtered):
    if 'MatchTime' in matches_filtered.columns:
        matches_filtered['MatchTime'] = pd.to_datetime(
            matches_filtered['MatchTime'], format='%H:%M:%S', errors='coerce'
        ).dt.time

        invalid_mask = matches_filtered['MatchTime'].isna()
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:

            def random_time():
                h = np.random.randint(0, 24)
                m = np.random.randint(0, 60)
                s = np.random.randint(0, 60)
                return datetime.time(h, m, s)

            matches_filtered.loc[invalid_mask, 'MatchTime'] = [random_time() for _ in range(num_invalid)]

    elo_cols = ["HomeElo", "AwayElo"]
    for col in elo_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()
            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value

    forms_3_cols = ["Form3Home", "Form3Away"]
    for col in forms_3_cols:
        if col in matches_filtered.columns:
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0) | (matches_filtered[col] > 9)
            num_invalid = invalid_mask.sum()
            mean_value = int(round(matches_filtered[col].mean(skipna=True)))
            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value
            matches_filtered[col] = matches_filtered[col].astype(int)


    forms_5_cols = ["Form5Home", "Form5Away"]
    for col in forms_5_cols:
        if col in matches_filtered.columns:
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0) | (matches_filtered[col] > 15)
            num_invalid = invalid_mask.sum()
            mean_value = int(round(matches_filtered[col].mean(skipna=True)))
            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value
            matches_filtered[col] = matches_filtered[col].astype(int)

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
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')

            mean_value = int(round(matches_filtered[col].mean(skipna=True)))

            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value

            matches_filtered[col] = matches_filtered[col].astype(int)

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
                matches_filtered.loc[invalid_mask, col] = mean_value

    handi_size_col = ["HandiSize"]

    for col in handi_size_col:
        if col in matches_filtered.columns:
            matches_filtered[col] = pd.to_numeric(matches_filtered[col], errors='coerce')

            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna()
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value

    handicap_cols = ["HandiHome", "HandiAway"]

    for col in handicap_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)
            invalid_mask = matches_filtered[col].isna() | (matches_filtered[col] < 0)
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value

    prob_cols = ['C_LTH', 'C_LTA', 'C_VHD', 'C_VAD', 'C_HTB', 'C_PHB']

    for col in prob_cols:
        if col in matches_filtered.columns:
            mean_value = matches_filtered[col].mean(skipna=True)

            invalid_mask = (matches_filtered[col].isna()) | (matches_filtered[col] < 0.0001) | (matches_filtered[col] > 0.9999)
            num_invalid = invalid_mask.sum()

            if num_invalid > 0:
                matches_filtered.loc[invalid_mask, col] = mean_value

    return matches_filtered


matches_before_clearing = matches_df.copy()

matches_after_cleaning = clean_data(matches_filtered)

matches_after_cleaning.to_csv("../csv/Exploratory_matches_cleaned.csv", index=False)

print("Dane zapisane do Exploratory_matches_cleaned.csv.csv")