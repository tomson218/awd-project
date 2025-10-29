from cleared_data import matches_after_cleaning
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def calculate_values(matches_after_cleaning: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = matches_after_cleaning.select_dtypes(include='number').columns # wybór jedynie kolumn numerycznych

    ranges = pd.DataFrame({
        'min': matches_after_cleaning[numeric_cols].min(),
        'max': matches_after_cleaning[numeric_cols].max(),
        "average": matches_after_cleaning[numeric_cols].mean(),
        'std': matches_after_cleaning[numeric_cols].std(),  # <-- tutaj dodane odchylenie standardowe
        'median': matches_after_cleaning[numeric_cols].median(),
        'Q1': matches_after_cleaning[numeric_cols].quantile(0.25),
        'Q3': matches_after_cleaning[numeric_cols].quantile(0.75),
        'quantile_0.1': matches_after_cleaning[numeric_cols].quantile(0.1),
        'quantile_0.9': matches_after_cleaning[numeric_cols].quantile(0.9)
    })

    return ranges

"""3. Wyznaczenie współczynnika korelacji liniowej Pearsona pomiędzy wybranymi parami
zmiennych (lub autokorelacji dla danych jednowymiarowych)."""
def pearson_correlation():
    # kodowanie
    HTResult = matches_after_cleaning["HTResult"]
    FTResult = matches_after_cleaning["FTResult"]

    mapping = {"H": 1, "D": 0, "A": -1}
    HT_num = [mapping[x] for x in HTResult]
    FT_num = [mapping[x] for x in FTResult]

    r, p_value = pearsonr(HT_num, FT_num)
    return r, p_value

"""4. Dokonanie prostej regresji liniowej pomiędzy wybranymi zmiennymi. Należy wyznaczyć
wzór regresji."""
def linear_regression():
    X = np.concatenate([
        matches_after_cleaning["HomeTarget"].values,
        matches_after_cleaning["AwayTarget"].values
    ]).reshape(-1, 1)  # -1 to dopasowanie pod liczbę wierszy, 1 - zdefiniowanie jednego wymiaru

    Y = np.concatenate([
        matches_after_cleaning["FTHome"].values,
        matches_after_cleaning["FTAway"].values
    ])

    # model regresji
    model = LinearRegression()
    model.fit(X, Y)

    a = model.coef_[0]
    b = model.intercept_

    return a, b


def descriptive_stats_all():
    print(calculate_values(matches_after_cleaning))

    #PEARSON_CORRELATION
    r, p_value = pearson_correlation()
    print(f"Współczynnik korelacji r (wpływ wyniku do przerwy na ostateczny wynik meczu) = {r:.2f}")
    #print(f"P-wartość = {p_value:.4f}")

    #LINEAR_REGRESSION 
    a, b = linear_regression()
    print(f"Wzór regresji: Y = {a:.2f} * X + {b:.2f}")