import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from cleared_data import matches_after_cleaning
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch

def pie_chart():
    divisions = matches_after_cleaning['Division'].unique()
    counts = []

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        unique_teams = set(df_div['HomeTeam']).union(set(df_div['AwayTeam']))
        counts.append(len(unique_teams))

    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie'
    }
    labels = [league_names.get(div, div) for div in divisions]

    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title("Procentowy wskażnik drużyn w poszczególnych ligach na przełomie sezonów 2022/2025")
    plt.show()


def bar_chart():
    divisions = matches_after_cleaning['Division'].unique()
    counts = []

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        unique_teams = set(df_div['HomeTeam']).union(set(df_div['AwayTeam']))
        counts.append(len(unique_teams))

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

    # Liczby na slupkach
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title("Liczba drużyn w każdej lidze w sezonach 2022–2025", fontsize=16, pad=20)
    plt.ylabel("Liczba drużyn", fontsize=12)
    plt.xlabel("Liga", fontsize=12)
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()


def bar_chart_cards_global():
    df = matches_after_cleaning.copy()

    avg_yellow_home = df['HomeYellow'].mean().round(2)
    avg_yellow_away = df['AwayYellow'].mean().round(2)
    avg_red_home = df['HomeRed'].mean().round(2)
    avg_red_away = df['AwayRed'].mean().round(2)

    labels = [
        'Żółte (gospodarze)',
        'Żółte (goście)',
        'Czerwone (gospodarze)',
        'Czerwone (goście)'
    ]
    values = [avg_yellow_home, avg_yellow_away, avg_red_home, avg_red_away]
    colors = ['#ffcc00', '#ffcc00', '#ff3333', '#ff3333']  # żółty i czerwony
    edge_colors = ['#cc9900', '#cc9900', '#cc0000', '#cc0000']

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        labels, values,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.5,
        width=0.6
    )

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f'{val}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='black'
        )

    # --- 4. Formatowanie ---
    plt.title('Średnia liczba kartek na mecz z podziałem na gospodarza i gościa',
              fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Średnia liczba kartek', fontsize=12)
    plt.ylim(0, max(values) * 1.3)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Legenda

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

    home = matches_after_cleaning[['MatchDate', 'HomeTeam', 'HomeElo']].copy()
    home = home.rename(columns={'HomeTeam': 'club', 'HomeElo': 'elo'})

    away = matches_after_cleaning[['MatchDate', 'AwayTeam', 'AwayElo']].copy()
    away = away.rename(columns={'AwayTeam': 'club', 'AwayElo': 'elo'})

    club_data = pd.concat([home, away], ignore_index=True)
    club_data = club_data[club_data['club'].isin(clubs_to_compare)]
    club_data['MatchDate'] = pd.to_datetime(club_data['MatchDate'])
    club_data = club_data.sort_values(['club', 'MatchDate'])

    plt.figure(figsize=(14, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    color_map = dict(zip(clubs_to_compare, colors))

    # Rysuj linię dla każdego klubu
    for club in clubs_to_compare:
        data = club_data[club_data['club'] == club]
        plt.plot(
            data['MatchDate'], data['elo'],
            label=club,
            color=color_map[club],
            markersize=6,
            linewidth=2,
            linestyle='-',
            alpha=0.9
        )

    plt.title('Szereg czasowy dla 5 najlepszych drużyn z każdej lig wg wskaźnika Elo', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Wskaźnik Elo', fontsize=12)
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
def histogram_home():
    goals = matches_after_cleaning['FTHome'].dropna().astype(int)

    plt.figure(figsize=(11, 6))
    n, bins, patches = plt.hist(
        goals,
        bins=range(0, goals.max() + 2),  # zostaje
        color='#2E86AB',
        edgecolor='black',
        alpha=0.8
    )

    plt.title("Histogram liczby goli strzelonych przez gospodarzy", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Liczba goli", fontsize=12)
    plt.ylabel("Liczba meczów", fontsize=12)

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, goals.max() + 1))
    plt.tight_layout()
    plt.show()


# --- 4. Histogram ---
def histogram_away():
    goals = matches_after_cleaning['FTAway'].dropna().astype(int)

    plt.figure(figsize=(11, 6))
    n, bins, patches = plt.hist(
        goals,
        bins=range(0, goals.max() + 2),  # zostaje
        color='#2E86AB',
        edgecolor='black',
        alpha=0.8
    )

    plt.title("Histogram liczby goli strzelonych przez gościa", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Liczba goli", fontsize=12)
    plt.ylabel("Liczba meczów", fontsize=12)

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, goals.max() + 1))
    plt.tight_layout()
    plt.show()


# --- 5. Wykres pudełkowy ---
def box_plot():
    divisions = matches_after_cleaning['Division'].unique()
    all_elo_ratings_data = [] 

    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie'
    }

    print("Statystyki wskaźnika Elo dla poszczególnych lig:\n")
    print(f"{'Liga':<12} {'Mediana':<10} {'Q1':<10} {'Q3':<10} {'Min':<10} {'Max':<10} {'IQR':<10}")
    print("-" * 70)

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]
        
        # Łączenie
        elo_ratings = df_div['HomeElo'].dropna().tolist() + df_div['AwayElo'].dropna().tolist()
        all_elo_ratings_data.append(elo_ratings)

    plt.figure(figsize=(12, 6))
    plt.boxplot(all_elo_ratings_data, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'))

    labels = [league_names.get(div, div) for div in divisions]
    plt.xticks(range(1, len(divisions) + 1), labels, rotation=45)

    plt.xlabel("Liga")
    plt.ylabel("Wskaźnik Elo")
    plt.title("Wykres pudełkowy dla wskaznika Elo wg poszczególnych lig")
    plt.show()

# --- 6. Wykres wiolinowy ---
def violin_plot():
    league_names = {
        'I1': 'Serie A',
        'SP1': 'La Liga',
        'D1': 'Bundesliga',
        'F1': 'Ligue 1',
        'N1': 'Eredivisie',
    }

    divisions = matches_after_cleaning['Division'].unique()
    all_fouls_data = []
    labels = []
    medians = []

    for div in divisions:
        df_div = matches_after_cleaning[matches_after_cleaning['Division'] == div]

        # Łączenie fauli
        fouls = pd.concat([
            df_div['HomeFouls'].dropna(),
            df_div['AwayFouls'].dropna()
        ]).astype(float)

        all_fouls_data.append(fouls.tolist())
        label = league_names.get(div, div)
        labels.append(label)
        medians.append(fouls.median())

    # --- 3. Sortowanie: od ligi
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

    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_edgecolor('darkred')
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('red')
    parts['cmedians'].set_linewidth(2.5)

    parts['cmins'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cbars'].set_color('black')


    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right', fontsize=11)
    plt.xlabel("Liga", fontsize=13)
    plt.ylabel("Liczba fauli w meczu", fontsize=13)
    plt.title("Wykres wiolinowy dla fauli dla drużyn pogrupowane według poszczególnych lig",
              fontsize=15, pad=20, fontweight='bold')

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    #return plt.gcf()

# --- 7. Wykres rozrzutu + regresja liniowa ---
def scatter_plot():
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

    axes[0].scatter(X_home, Y_home, color='blue', alpha=0.6, label='Dane')
    axes[0].plot(X_home, model_home.predict(X_home), color='red', label=f'y = {a_home:.2f}x + {b_home:.2f}')
    axes[0].set_title("Regresja: Strzałów celnych vs Liczba goli strzelonych w meczu przez gospodarza")
    axes[0].set_xlabel("Strzały celne gospodarza")
    axes[0].set_ylabel("Liczba bramek gospodarza strzelona w meczu")
    axes[0].legend()

    # --- AWAY ---
    axes[1].scatter(X_away, Y_away, color='green', alpha=0.6, label='Dane')
    axes[1].plot(X_away, model_away.predict(X_away), color='red', label=f'y = {a_away:.2f}x + {b_away:.2f}')
    axes[1].set_title("Regresja: Strzałów celnych vs Liczba goli strzelonych w meczu przez gościa")
    axes[1].set_xlabel("Strzały celne gościa")
    axes[1].set_ylabel("Liczba bramek gościa strzelona w meczu")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return (a_home, b_home), (a_away, b_away)


def visualizations_all():
    pie_chart()
    bar_chart()
    bar_chart_cards_global()
    time_series_plot()
    histogram_home()
    histogram_away()
    box_plot()
    violin_plot()
    scatter_plot()
