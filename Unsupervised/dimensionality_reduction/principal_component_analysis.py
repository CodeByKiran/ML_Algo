from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")
target_cols = ['Churn_Yes','Churn_num']
X = df.drop(columns=target_cols)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X)

print("Original number of features:", X.shape[1])
print("Reduced number of features after PCA:", X_pca.shape[1])
