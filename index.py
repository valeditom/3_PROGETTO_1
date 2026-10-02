import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =========================
# 1. CARICAMENTO DATASET
# =========================

iris = load_iris()
X = iris.data
df = pd.DataFrame(X, columns=iris.feature_names)

print(df.head())
print(df.describe())

# =========================
# STANDARDIZZAZIONE
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 2. DBSCAN
# =========================

eps_values = [0.35, 0.45, 0.55]

for eps in eps_values:
    dbscan = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels_dbscan = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels_dbscan)) - (
        1 if -1 in labels_dbscan else 0
    )

    n_noise = list(labels_dbscan).count(-1)

    print(f"\nEPS: {eps}")
    print(f"Numero cluster: {n_clusters}")
    print(f"Punti rumorosi: {n_noise}")