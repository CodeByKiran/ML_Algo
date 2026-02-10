import pandas as pd 
import numpy as np

from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder

df = pd.read_csv(r"C:\DATA_SCIENCE\data_set\TelCom_churn_preprocessed.csv")

# select Feature Columns  
X = df[['SeniorCitizen', 'tenure', 'MonthlyCharges','TotalCharges',
       'gender_Male', 'Partner_Yes', 'Dependents_Yes',
       'PhoneService_Yes', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']]

#target Feature
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf  = RandomForestClassifier(
    n_estimators = 200,
    max_depth = None,
    criterion ='gini',
    min_samples_split = 2,
    min_samples_leaf = 1,
    random_state = 42,
    n_jobs = -1
)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)


print("Accuracy :",accuracy_score(y_test,y_pred))
print("\nConfusion Matrix :\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report :\n",classification_report(y_test,y_pred))