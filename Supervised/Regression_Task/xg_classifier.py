import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score,roc_auc_score,RocCurveDisplay,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")

X = df.drop(columns= ["Churn_Yes","Churn_num"])
y = df["Churn_Yes"]

X_train , X_test , y_train,y_test = train_test_split(
    X,y,test_size = 0.2 , random_state = 42 , stratify=y
)


from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 5,
    random_state = 42,
    n_jobs = -1
)

xgb.fit(X_train,y_train)

y_pred = xgb.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix :\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report: \n",classification_report(y_test,y_pred))

