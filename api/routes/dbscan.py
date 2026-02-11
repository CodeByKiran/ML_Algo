from fastapi import APIRouter
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

router = APIRouter(
    prefix="/dbscan",
    tags=["DBSCAN"]
)

@router.get("/")
def run_dbscan():
    # Load data
    df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")

    target_cols = ['Churn_Yes', 'Churn_Num']
    X = df.drop(columns=target_cols)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    df["dbscan_cluster"] = labels

    # Metrics
    cluster_counts = df["dbscan_cluster"].value_counts().to_dict()

    churn_dist = (
        df.groupby("dbscan_cluster")["Churn_Num"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .to_dict()
    )

    silhouette = None
    mask = labels != -1
    if len(set(labels)) > 1 and mask.sum() > 1:
        silhouette = silhouette_score(X_scaled[mask], labels[mask])

    return {
        "algorithm": "DBSCAN",
        "eps": 2,
        "min_samples": 5,
        "cluster_counts": cluster_counts,
        "churn_distribution": churn_dist,
        "silhouette_score": silhouette
    }
