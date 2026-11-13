# ⚽ AWD Project — Analiza danych meczów piłki nożnej

Projekt do analizy danych meczów piłki nożnej z 5 największych europejskich lig: **Serie A**, **La Liga**, **Bundesliga**, **Ligue 1** i **Eredivisie**. Składa się z dwóch głównych modułów: **analiza statystyczna** (eksploracja i wizualizacja danych) oraz **analiza eksploracyjna z ML** (predykcja wyników meczów).

---

## 📊 Dane wejściowe

| Cecha | Opis |
|-------|------|
| **Źródło** | [Kaggle: Club Football Match Data 2000-2025](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) |
| **Pliki** | `Matches.csv` (dane meczów), `EloRatings.csv` (ratingi Elo) |
| **Cechy** | Wyniki HT/FT, Elo, forma (Form3/Form5), strzały, faule, rożne, karty, kursy bukmacherskie, handikapy, prawdopodobieństwa |
| **Filtr lig** | I1 (Serie A), SP1 (La Liga), D1 (Bundesliga), F1 (Ligue 1), N1 (Eredivisie) |

---

## 🛠 Stack technologiczny

- **Język:** Python 3.12+
- **Przetwarzanie danych:** `pandas`, `numpy`
- **Statystyka:** `scipy.stats` (korelacja Pearsona)
- **Wizualizacja:** `matplotlib`, `seaborn`, `plotly`
- **Machine Learning:** `scikit-learn` (regresja liniowa, skalowanie, podział danych), `catboost` (klasyfikator gradient boosting)
- **Serializacja:** `joblib` (zapis scalerów)
- **Pobieranie danych:** `kagglehub`

---

## 📁 Struktura projektu

```
awd-project/
├── requirements.txt              # Zależności Pythona
├── csv/                          # Oczyszczone dane CSV
│   └── Exploratory_matches_cleaned.csv
├── statistical_analysis/         # Moduł analizy statystycznej
│   ├── main.py                   # Punkt wejścia
│   ├── cleared_data.py           # ETL: pobieranie, filtrowanie, czyszczenie danych
│   ├── descriptive_stats.py      # Statystyki opisowe, korelacja, regresja
│   └── visualisation.py          # 9 wizualizacji danych
└── exploratory_analysis/         # Moduł analizy eksploracyjnej z ML
    ├── prepare_data.py           # ETL do ML (dane 2020-2025)
    ├── data_processing.py        # Przygotowanie cech i zmiennych docelowych
    ├── classification_data.py    # Trenowanie modelu CatBoost
    └── testing.py                # Testowanie i ocena modelu
```

---

## 🚀 Instalacja

### Wymagania systemowe

- Python 3.12 lub nowszy
- Kaggle API (do pobrania danych)

### Kroki instalacji

1. **Sklonuj repozytorium:**
   ```bash
   git clone <repo-url>
   cd awd-project
   ```

2. **Stwórz wirtualne środowisko:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Linux/Mac
   ```

3. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Skonfiguruj Kaggle API:**
   - Pobierz plik `kaggle.json` z [Kaggle Account → API](https://www.kaggle.com/account)
   - Umieść go w folderze `~/.kaggle/`

---

## ▶️ Uruchamianie

### Moduł 1: Analiza statystyczna

Analiza danych z lat **2022–2025**. Uruchamia czyszczenie danych, statystyki opisowe i 9 wizualizacji.

```bash
cd statistical_analysis
python main.py
```

**Co robi:**
1. **`cleared_data.py`** — Pobiera dane z Kaggle, filtruje do 5 top lig i lat 2022-2025, czyści brakujące wartości (uzupełnia średnimi), zapisuje do `Matches_cleaned.csv`
2. **`descriptive_stats.py`** — Oblicza statystyki opisowe (min, max, średnia, odchylenie, mediana, kwartyle), korelację Pearsona (wpływ HTResult na FTResult), regresję liniową (strzały celne → gole), macierz korelacji (zapis do Excela)
3. **`visualisation.py`** — Generuje 9 wizualizacji:
   - 🥧 **Pie chart** — procentowy udział drużyn w poszczególnych ligach
   - 📊 **Bar chart** — liczba drużyn w każdej lidze
   - 📊 **Bar chart kart** — średnia liczba żółtych i czerwonych kart (gospodarze vs goście)
   - 📈 **Szereg czasowy** — wskaźnik Elo dla 5 klubów (Real Madrid, Ajax, Bayern, Inter, Paris SG)
   - 📊 **Histogram** — rozkład goli gospodarzy
   - 📊 **Histogram** — rozkład goli gości
   - 📦 **Box plot** — rozkład Elo wg lig
   - 🎻 **Violin plot** — rozkład fauli wg lig
   - 🔵 **Scatter plot + regresja** — strzały celne vs gole (gospodarze i goście)

### Moduł 2: Analiza eksploracyjna z ML

Predykcja wyników meczów na podstawie danych z lat **2020–2025**.

```bash
cd exploratory_analysis

# Krok 1: Pobranie i oczyszczenie danych
python prepare_data.py

# Krok 2: Przygotowanie cech i zmiennych docelowych
python data_processing.py

# Krok 3: Trenowanie modelu CatBoost z Grid Search
python classification_data.py

# Krok 4: Testowanie i ocena modelu
python testing.py
```

**Co robi każdy krok:**

1. **`prepare_data.py`** — Analogiczny ETL do `cleared_data.py`, ale szerszy zakres dat (2020–2025). Zapisuje do `csv/Exploratory_matches_cleaned.csv`

2. **`data_processing.py`** — Wybiera 25 cech numerycznych (Elo, Form3/5, kursy, handikapy, prawdopodobieństwa), normalizuje `MinMaxScaler` do [0, 1], tworzy zmienne binarne (`HomeNoLose`, `AwayWin`). Zapisuje: `X_scaled.csv`, `y_binary.csv`, `scaler.pkl`

3. **`classification_data.py`** — Trenuje **CatBoost Classifier** z zadaniem klasyfikacji binarnej (czy gospodarz nie przegra). Podział: 70% train, 10% validation, 20% test (stratified). **Grid Search** z 5-fold cross-validation, optymalizacja AUC. Metryki: AUC, Accuracy, Precision, Recall, Logloss. Zapisuje model do `models/catboost_home_no_lose.cbm` i wyniki do `results/binary_results.json`

4. **`testing.py`** — Ładuje wytrenowany model, oblicza metryki testowe (Accuracy, F1, ROC AUC, LogLoss), generuje macierz pomyłek (Seaborn heatmap) i 20 losowych przykładów z porównaniem przewidywań z rzeczywistymi wynikami. Zapisuje wyniki do CSV

---

## 📈 Kluczowe wnioski z analizy

- **Korelacja HT→FT:** Wynik do przerwy ma silny wpływ na ostateczny wynik meczu (wysoki współczynnik korelacji Pearsona)
- **Strzały celne → gole:** Regresja liniowa potwierdza zależność między strzałami celnymi a liczbą strzelonych goli
- **Najważniejsze cechy do predykcji:** Kursy bukmacherskie (`MaxHome` — 14.4%, `MaxAway` — 10.3%, `OddAway` — 9.5%) i handikapy — co jest intuicyjne, ponieważ kursy agregują wiedzę rynkową
- **Model CatBoost:** Osiąga dobre wyniki w predykcji, czy gospodarz nie przegra (HomeNoLose)

---

## 🔄 Przepływ pracy

```
Kaggle → Pobierz dane → Filtr lig/dat → Czyszczenie → 
├── statistical_analysis: Statystyki + 9 wizualizacji
└── exploratory_analysis: Normalizacja → CatBoost → Grid Search → Test → Wyniki
```
