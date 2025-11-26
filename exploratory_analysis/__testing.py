# 1. Predykcja

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from __classification_data import best_params

