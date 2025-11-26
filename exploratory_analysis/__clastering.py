""""# clustering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === KONFIGURACJA ===
# 1. Ścieżka do folderu z danymi
DATA_DIR = os.path.join("..", "data")  # <-- W górę o jeden poziom, potem do /data
X_FILE = os.path.join(DATA_DIR, "X_processed.csv")
ORIGINAL_FILE = "../csv/Matches_cleaned.csv"  # <-- ZMIEŃ JEŚLI TRZEBA
numeric_cols = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']

# Utwórz folder na wyniki
os.makedirs(DATA_DIR, exist_ok=True)

print("Wczytywanie danych...")

# === 1. WCZYTAJ PRZETWORZONE CECHY X ===
X_processed = pd.read_csv(X_FILE)
print(f"X_processed wczytano: {X_processed.shape}")

# === 2. WCZYTAJ ORYGINALNE DANE (żeby mieć FTResult) ===
df_original = pd.read_csv(ORIGINAL_FILE)
if 'FTResult' not in df_original.columns:
    raise ValueError("Brak kolumny 'FTResult' w pliku Matches_cleaned.csv")
print(f"Oryginalne dane: {df_original.shape}")

# Sprawdź, czy liczba wierszy się zgadza
if len(X_processed) != len(df_original):
    print(f"OSTRZEŻENIE: X_processed ({len(X_processed)}) ≠ df_original ({len(df_original)})")
    min_len = min(len(X_processed), len(df_original))
    X_processed = X_processed.iloc[:min_len]
    df_original = df_original.iloc[:min_len]
    print(f"Przycięto do {min_len} wierszy.")

# === 3. METODA ŁOKCIA ===
print("\nDobór liczby klastrów (Elbow Method)...")
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.title('Metoda Łokcia - Dobór K')
plt.xlabel('Liczba klastrów K')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, 'elbow_method.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Zapisano: data/elbow_method.png")

# === 4. KLASTERYZACJA (wybierz K ręcznie po wykresie!) ===
optimal_k = 4  # ZMIEŃ PO ZOBACZENIU WYKRESU ŁOKCIA
print(f"\nUruchamiam K-Means z K = {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)

# Dodaj klastry do danych
df = df_original.copy()
df['Cluster'] = cluster_labels

# === 5. PCA - WIZUALIZACJA 2D ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=cluster_labels, cmap='tab10', alpha=0.7, s=50, edgecolor='k'
)
plt.title(f'Klasteryzacja K-Means (K={optimal_k}) - PCA')
plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.1%} wariancji)')
plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.1%} wariancji)')
plt.colorbar(scatter, label='Klaster')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(DATA_DIR, 'clusters_pca.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Zapisano: data/clusters_pca.png")

# === 6. ANALIZA: ROZKŁAD WYNIKÓW W KAŻDYM KLASTRZE ===
print("\nRozkład wyników (H/D/A) w klastrach:")
result_dist = pd.crosstab(df['Cluster'], df['FTResult'], normalize='index')
result_dist = result_dist.reindex(columns=['H', 'D', 'A'], fill_value=0)
print(result_dist.round(3))
result_dist.to_csv(os.path.join(DATA_DIR, "cluster_result_distribution.csv"))
print("Zapisano: data/cluster_result_distribution.csv")

# === 7. ŚREDNIE CECH W KAŻDYM KLASTRZE ===
X_with_cluster = X_processed.copy()
X_with_cluster['Cluster'] = cluster_labels
cluster_means = X_with_cluster.groupby('Cluster').mean()
print("\nŚrednie cechy w klastrach [0,1]:")
print(cluster_means.round(3))
cluster_means.to_csv(os.path.join(DATA_DIR, "cluster_feature_means.csv"))
print("Zapisano: data/cluster_feature_means.csv")

# === 8. ZAPIS CAŁYCH DANYCH Z KLASTRAMI ===
df.to_csv(os.path.join(DATA_DIR, "matches_with_clusters.csv"), index=False)
print("Zapisano: data/matches_with_clusters.csv")

# === 9. CENTROIDY ===
centroids_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
centroids_scaled.to_csv(os.path.join(DATA_DIR, "cluster_centroids_scaled.csv"), index=False)
print("Zapisano: data/cluster_centroids_scaled.csv")

# === 10. AUTOMATYCZNE NAZWY KLASTRÓW (na podstawie różnic w Elo i formie) ===
def nazwij_klaster(row):
    home_elo = row['HomeElo']
    away_elo = row['AwayElo']
    home_form = (row['Form3Home'] + row['Form5Home']) / 2
    away_form = (row['Form3Away'] + row['Form5Away']) / 2

    elo_diff = home_elo - away_elo
    form_diff = home_form - away_form

    if elo_diff > 0.15 and home_form > 0.6:
        return "Silny Gospodarz"
    elif elo_diff < -0.15 and away_form > 0.6:
        return "Silny Gość"
    elif abs(elo_diff) < 0.1 and abs(form_diff) < 0.2:
        return "Wyrównany"
    elif elo_diff > 0.1:
        return "Lekka Przewaga Gospodarza"
    elif elo_diff < -0.1:
        return "Lekka Przewaga Gościa"
    else:
        return "Nieokreślony"

centroids_named = centroids_scaled.copy()
centroids_named['Nazwa_Klastra'] = centroids_scaled.apply(nazwij_klaster, axis=1)
print("\nProponowane nazwy klastrów:")
print(centroids_named[['Nazwa_Klastra']])

# Zapis z nazwami
centroids_named.to_csv(os.path.join(DATA_DIR, "cluster_centroids_with_names.csv"), index=False)
print("Zapisano: data/cluster_centroids_with_names.csv")

# Dodaj nazwy do df
df['Cluster_Name'] = df['Cluster'].map(
    centroids_named.set_index(centroids_named.index)['Nazwa_Klastra']
)
df[['Cluster', 'Cluster_Name'] + numeric_cols + ['FTResult']].head(10)

print(f"\nKlasteryzacja zakończona! K = {optimal_k}")
print(f"Rozkład klastrów:\n{pd.Series(cluster_labels).value_counts().sort_index()}")
"""