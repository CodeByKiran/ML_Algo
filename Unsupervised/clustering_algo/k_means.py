
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")


target_cols = ['Churn_Yes', 'Churn_Num']
X = df.drop(columns=target_cols)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


X = df.drop('Churn', axis=1)


wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(2,11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X)


print("\nCluster Counts:")
print(df['kmeans_cluster'].value_counts())


print("\nChurn Distribution per Cluster:")
print(df.groupby('kmeans_cluster')['Churn'].value_counts(normalize=True))


plt.figure(figsize=(8,5))
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=df['kmeans_cluster'], palette='Set1', s=60)
plt.title("K-Means Clusters (2 Features)")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.show()


sns.pairplot(df, vars=X.columns[:4], hue='kmeans_cluster', palette='Set1')
plt.show() '''

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_kmeans(data, k):
    """
    data: List[List[float]]
    k: int
    """

    X = np.array(data)

    # Optional scaling (recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)

    return {
        "labels": labels.tolist(),
        "centroids": model.cluster_centers_.tolist()
    }
