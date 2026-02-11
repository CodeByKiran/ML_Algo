from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from  sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd




df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")
target_cols =['Churn_Yes','Churn_num']
X = df.drop(columns=target_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=2, min_samples=5)  
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)  


print("DBSCAN Cluster Counts:")
print(df['dbscan_cluster'].value_counts())

print("\nChurn Distribution per DBSCAN Cluster:")
print(df.groupby('dbscan_cluster')['Churn_num'].value_counts(normalize=True))


mask = df['dbscan_cluster'] != -1
if len(df['dbscan_cluster'].unique()) > 1:
    score = silhouette_score(X[mask], df['dbscan_cluster'][mask])
    print(f"\nSilhouette Score (excluding noise): {score:.3f}")


plt.figure(figsize=(8,5))
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df['dbscan_cluster'], palette='Set1', s=60)
plt.title("DBSCAN Clusters (2 Features)")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.show()


'''

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

def run_dbscan(data, eps=0.5, min_samples=5):
    """
    data: List[List[float]]
    eps: float
    min_samples: int
    """

    X = np.array(data)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    # Metrics
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)

    cluster_distribution = {
        int(label): int(list(labels).count(label))
        for label in unique_labels if label != -1
    }

    # Silhouette Score (only if valid)
    silhouette = None
    mask = labels != -1
    if n_clusters > 1 and mask.sum() > n_clusters:
        silhouette = float(
            silhouette_score(X_scaled[mask], labels[mask])
        )

    return {
        "n_clusters": n_clusters,
        "noise_points": noise_points,
        "cluster_distribution": cluster_distribution,
        "silhouette_score": silhouette
    }
'''