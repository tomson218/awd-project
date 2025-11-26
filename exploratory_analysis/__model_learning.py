"""import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ================================
# 1. WCZYTANIE
# ================================
data_dir = '../data/splits'
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).values
X_test  = pd.read_csv(os.path.join(data_dir, 'X_test.csv')).values
y_train_onehot = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values
y_test_onehot  = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values

# Konwersja one-hot → etykiety
y_train = np.argmax(y_train_onehot, axis=1)
y_test  = np.argmax(y_test_onehot, axis=1)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
"""
