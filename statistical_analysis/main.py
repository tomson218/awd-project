from cleared_data import matches_before_clearing, matches_after_cleaning
from descriptive_stats import descriptive_stats_all
from visualisation import visualizations_all

print ("Matches data before clearing:", matches_before_clearing.head())

print("Czy są już czyste: ", matches_before_clearing.isnull().sum())


print ("Data cleared successfully\n.", matches_after_cleaning.head())

print("Czy są już czyste: ", matches_after_cleaning.isnull().sum())

print("Liczba rekordów:", len(matches_after_cleaning))


print(descriptive_stats_all())

print(visualizations_all())