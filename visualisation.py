"""5. Wizualizację wybranych danych w postaci:
• wykresów kołowych, słupkowych prezentujących liczebność rekordów w
poszczególnych klasach (kategoriach),
• wykresów prezentujących szeregi czasowe (dla predykcji),
• histogramy zmiennych,
• wykresów pudełkowych,
• wykresów wiolinowych,
• wykresów rozrzutu dla wybranych par zmiennych oraz regresji liniowej między nimi.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from cleared_data import matches_after_cleaning
from sklearn.linear_model import LinearRegression

def pie_chart():
    """
    Wykres kołowy: udział każdej ligi w liczbie unikalnych drużyn.
    Etykiety: TYLKO procenty (np. "25.0%").
    """
    divisions = matches_after_cleaning['Division'].unique()
    counts = []

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        unique_teams = set(df_div['HomeTeam']).union(set(df_div['AwayTeam']))
        counts.append(len(unique_teams))

    # Mapowanie kodów na pełne nazwy
    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie'
    }
    labels = [league_names.get(div, div) for div in divisions]

    # 3. Najprostszy wykres kołowy
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title("Udział lig w liczbie unikalnych drużyn")
    plt.show()


def bar_chart():
    divisions = matches_after_cleaning['Division'].unique()
    counts = []

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        unique_teams = set(df_div['HomeTeam']).union(set(df_div['AwayTeam']))
        counts.append(len(unique_teams))

    # Mapowanie kodów lig na pełne nazwy (opcjonalnie)
    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie'
    }
    labels = [league_names.get(div, div) for div in divisions]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['#9467bd', '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e'])

    # Dodaj liczby na słupkach
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Liczba unikalnych drużyn w każdej lidze (2022–2025)", fontsize=16, pad=20)
    plt.ylabel("Liczba drużyn", fontsize=12)
    plt.xlabel("Liga", fontsize=12)
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


# --- Przykład użycia ---
#fig = bar_chart()
#plt.show()

def bar_chart_cards_global():
    df = matches_after_cleaning.copy()

    # --- 1. Oblicz średnie ---
    avg_yellow_home = df['HomeYellow'].mean().round(2)
    avg_yellow_away = df['AwayYellow'].mean().round(2)
    avg_red_home = df['HomeRed'].mean().round(2)
    avg_red_away = df['AwayRed'].mean().round(2)

    # --- 2. Dane do wykresu ---
    labels = [
        'Żółte\n(gospodarze)',
        'Żółte\n(goście)',
        'Czerwone\n(gospodarze)',
        'Czerwone\n(goście)'
    ]
    values = [avg_yellow_home, avg_yellow_away, avg_red_home, avg_red_away]
    colors = ['#ffcc00', '#ffcc00', '#ff3333', '#ff3333']  # żółty i czerwony
    edge_colors = ['#cc9900', '#cc9900', '#cc0000', '#cc0000']

    # --- 3. Wykres ---
    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        labels, values,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.5,
        width=0.6
    )

    # Etykiety na słupkach
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f'{val}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='black'
        )

    # --- 4. Formatowanie ---
    plt.title('Średnia liczba kartek na mecz\n(gospodarze vs goście)',
              fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Średnia liczba kartek', fontsize=12)
    plt.ylim(0, max(values) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ffcc00', edgecolor='#cc9900', label='Żółta kartka'),
        Patch(facecolor='#ff3333', edgecolor='#cc0000', label='Czerwona kartka')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.show()


# --- 3. Szereg czasowy (dla predykcji) ---
def time_series_plot():
    clubs_to_compare = ['Real Madrid', 'Ajax', 'Bayern Munich', 'Inter', 'Paris SG']

    # --- 1. Przygotuj dane ---
    home = matches_after_cleaning[['MatchDate', 'HomeTeam', 'HomeElo']].copy()
    home = home.rename(columns={'HomeTeam': 'club', 'HomeElo': 'elo'})

    away = matches_after_cleaning[['MatchDate', 'AwayTeam', 'AwayElo']].copy()
    away = away.rename(columns={'AwayTeam': 'club', 'AwayElo': 'elo'})

    club_data = pd.concat([home, away], ignore_index=True)
    club_data = club_data[club_data['club'].isin(clubs_to_compare)]
    club_data['MatchDate'] = pd.to_datetime(club_data['MatchDate'])
    club_data = club_data.sort_values(['club', 'MatchDate'])

    # --- 2. Rysuj wykres ręcznie ---
    plt.figure(figsize=(14, 8))

    # Kolory i markery dla każdego klubu
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    color_map = dict(zip(clubs_to_compare, colors))
    #marker_map = dict(zip(clubs_to_compare, markers))

    # Rysuj linię dla każdego klubu
    for club in clubs_to_compare:
        data = club_data[club_data['club'] == club]
        plt.plot(
            data['MatchDate'], data['elo'],
            label=club,
            color=color_map[club],
            #marker=marker_map[club],
            markersize=6,
            linewidth=2,
            linestyle='-',
            alpha=0.9
        )

    # --- 3. Formatowanie ---
    plt.title('Trendy Elo dla wybranych klubów (dom + wyjazd)', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Rating Elo', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    plt.legend(
        title='Klub',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    plt.tight_layout()
    plt.show()


# --- 4. Histogram ---
def histogram():
    """
        Podstawowy histogram bramek (FTHome) – TYLKO matplotlib.
        Bez seaborn, bez KDE, bez bajerów.
        """
    if 'FTHome' not in matches_after_cleaning.columns:
        print("Błąd: brak kolumny 'FTHome'")
        return

    goals = matches_after_cleaning['FTHome'].dropna().astype(int)

    plt.figure(figsize=(10, 6))

    # Podstawowy histogram
    counts, bins_edges, patches = plt.hist(
        goals,
        bins=10,
        color='#1f77b4',
        edgecolor='black',
        alpha=0.7
    )

    plt.title("Histogram bramek drużyny domowej (FTHome)",
              fontsize=16, fontweight='bold')
    plt.xlabel("Liczba bramek", fontsize=12)
    plt.ylabel("Liczba meczów", fontsize=12)

    bins = range(int(goals.min()), int(goals.max()) + 2)  # każdy słupek = 1 gol

    # Dodaj liczby na słupkach
    for count, bin_edge in zip(counts, bins_edges):
        if count > 0:
            plt.text(bin_edge + (bins_edges[1] - bins_edges[0]) / 2,
                     count + max(counts) * 0.01,
                     f'{int(count)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, int(goals.max()) + 1))
    plt.show()

# --- 5. Wykres pudełkowy (box plot) ---
def box_plot():
    divisions = matches_after_cleaning['Division'].unique()
    all_elo_ratings_data = []  # POPRAWKA: zdefiniuj listę!

    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie'
    }

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        # Łączymy Elo domowe i wyjazdowe
        elo_ratings = df_div['HomeElo'].dropna().tolist() + df_div['AwayElo'].dropna().tolist()
        all_elo_ratings_data.append(elo_ratings)  # teraz działa

    plt.figure(figsize=(12, 6))
    plt.boxplot(all_elo_ratings_data, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))

    plt.xticks(range(1, len(divisions) + 1), divisions, rotation=45)
    plt.xlabel("Dywizja")
    plt.ylabel("Elo Rating")  # POPRAWKA: nie "Liczba goli"
    plt.title("Rozkład Elo (Home + Away) wg Division")  # POPRAWKA
    plt.show()

# --- 6. Wykres wiolinowy ---
def violin_plot():
    # --- 1. Mapowanie kodów lig na pełne nazwy ---
    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie',
    }

    # --- 2. Przygotowanie danych ---
    divisions = matches_after_cleaning['Division'].unique()
    all_fouls_data = []
    labels = []
    medians = []  # do sortowania

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]

        # Łączymy faule: HomeFouls + AwayFouls (z każdego meczu)
        fouls = pd.concat([
            df_div['HomeFouls'].dropna(),
            df_div['AwayFouls'].dropna()
        ]).astype(float)

        all_fouls_data.append(fouls.tolist())
        label = league_names.get(div, div)
        labels.append(label)
        medians.append(fouls.median())

    # --- 3. Sortowanie: od ligi z NAJWIĘCEJ fauli (najwyższa mediana) ---
    sorted_indices = np.argsort(medians)[::-1]
    all_fouls_data = [all_fouls_data[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # --- 4. Rysowanie wykresu wiolinowego ---
    plt.figure(figsize=(14, 8))
    parts = plt.violinplot(
        all_fouls_data,
        showmeans=False,
        showmedians=True,
        showextrema=True
    )

    # Kolorowanie wiolin
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_edgecolor('darkred')
        pc.set_alpha(0.7)

    # Mediana – gruba, czerwona linia
    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(2.5)

    # Wąsy (min/max)
    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cbars'].set_color('black')

    # --- 5. Etykiety i formatowanie ---
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right', fontsize=11)
    plt.xlabel("Liga", fontsize=13)
    plt.ylabel("Liczba fauli w meczu", fontsize=13)
    plt.title("Rozkład fauli (Home + Away) w meczu – posortowane od najbrudniejszej ligi",
              fontsize=15, pad=20, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    return plt.gcf()

# --- 7. Wykres rozrzutu + regresja liniowa ---
def scatter_plot():
    # dane
    X_home = matches_after_cleaning["HomeTarget"].values.reshape(-1, 1)
    Y_home = matches_after_cleaning["FTHome"].values
    X_away = matches_after_cleaning["AwayTarget"].values.reshape(-1, 1)
    Y_away = matches_after_cleaning["FTAway"].values

    # regresje
    model_home = LinearRegression().fit(X_home, Y_home)
    model_away = LinearRegression().fit(X_away, Y_away)

    # współczynniki
    a_home, b_home = model_home.coef_[0], model_home.intercept_
    a_away, b_away = model_away.coef_[0], model_away.intercept_

    # tworzymy dwa wykresy obok siebie
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- HOME ---
    axes[0].scatter(X_home, Y_home, color='blue', alpha=0.6, label='Dane')
    axes[0].plot(X_home, model_home.predict(X_home), color='red', label=f'y = {a_home:.2f}x + {b_home:.2f}')
    axes[0].set_title("Regresja: HomeTarget vs FTHome")
    axes[0].set_xlabel("HomeTarget")
    axes[0].set_ylabel("FTHome")
    axes[0].legend()

    # --- AWAY ---
    axes[1].scatter(X_away, Y_away, color='green', alpha=0.6, label='Dane')
    axes[1].plot(X_away, model_away.predict(X_away), color='red', label=f'y = {a_away:.2f}x + {b_away:.2f}')
    axes[1].set_title("Regresja: AwayTarget vs FTAway")
    axes[1].set_xlabel("AwayTarget")
    axes[1].set_ylabel("FTAway")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return (a_home, b_home), (a_away, b_away)


def visualizations_all():
    pie_chart()
    bar_chart()
    bar_chart_cards_global()
    time_series_plot()
    histogram()
    box_plot()
    violin_plot()
    scatter_plot()
