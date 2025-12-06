from cleared_data import matches_after_cleaning
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def calculate_values(matches_after_cleaning: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = matches_after_cleaning.select_dtypes(include='number').columns

    ranges = pd.DataFrame({
        'Minimum': matches_after_cleaning[numeric_cols].min(),
        'Maksimum': matches_after_cleaning[numeric_cols].max(),
        "Średnia": matches_after_cleaning[numeric_cols].mean(),
        'Odchylenie': matches_after_cleaning[numeric_cols].std(),
        'Mediana': matches_after_cleaning[numeric_cols].median(),
        'Kwartyl Q1': matches_after_cleaning[numeric_cols].quantile(0.25),
        'Kwartyl Q3': matches_after_cleaning[numeric_cols].quantile(0.75),
        'Kwantyl_0.1': matches_after_cleaning[numeric_cols].quantile(0.1),
        'Kwantyl_0.9': matches_after_cleaning[numeric_cols].quantile(0.9)
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


def correlation_matrix(matches_after_cleaning: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza i zwraca macierz korelacji Pearsona dla wszystkich
    zmiennych numerycznych w DataFrame.
    """
    # 1. Wybierz tylko kolumny numeryczne
    numeric_df = matches_after_cleaning.select_dtypes(include=np.number)

    # 2. Oblicz macierz korelacji Pearsona
    # Domyślna metoda w .corr() to 'pearson'
    correlation_matrix = numeric_df.corr()

    return correlation_matrix

def descriptive_stats_all():
    print(calculate_values(matches_after_cleaning))

    r, p_value = pearson_correlation()
    print(f"Współczynnik korelacji r (wpływ wyniku do przerwy na ostateczny wynik meczu) = {r:.2f}")

    a, b = linear_regression()
    print(f"Wzór regresji: Y = {a:.2f} * X + {b:.2f}")

    matrix = correlation_matrix(matches_after_cleaning)
    print("Macierz korelacji:\n")
    print(matrix)