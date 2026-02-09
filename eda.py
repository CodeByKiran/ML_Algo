import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style ="whitegrid")

df = pd.read_csv("./data_set/TelCom_churn.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.describe(include ='str'))


print(df.isnull().sum())
print(df['TotalCharges'].unique()[:100])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors ='coerce')
print(df['TotalCharges'])

df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

print(df.isnull().sum())
print(df['Churn'].value_counts(normalize=True))

sns.countplot(x='Churn',data = df)
plt.title("Churn Distribution")
#plt.show()

print("x----x------x-------x-----x----Numerical Columns Exploration ----x------x------x----")
num_cols = ['tenure','MonthlyCharges','TotalCharges']


print("skewness of the Numerical Column's :\n" ,df[['tenure','MonthlyCharges','TotalCharges']].skew())
print("Co-relation  of the Numerical Column's :\n" ,df[['tenure','MonthlyCharges','TotalCharges']].corr(),'\n')

print(df[num_cols].describe())
plt.figure(figsize=(6,4))
sns.histplot(df['tenure'] , bins = 30 , kde=True)
plt.title("Tenure Distribution")
#plt.show()


plt.figure(figsize=(6,4))
sns.histplot(df['MonthlyCharges'],bins = 30, kde = True)
plt.title("Monthly Charges Distribution")
#plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['TotalCharges'] , bins =30 , kde=True)
plt.title("Total Charges Distribution")
#plt.show()


plt.figure(figsize=(12,4))  #Outliers Box plot 
plt.subplot(1,3,1)  #1
sns.boxplot(y=df['tenure'])
plt.title("Tenure")

plt.subplot(1,3,2) #2
sns.boxplot(y=df['MonthlyCharges'])
plt.title("Monthly Charges")

plt.subplot(1,3,1)  #3
sns.boxplot(y=df['TotalCharges'])
plt.title("Total Charges")
#plt.show()

sns.boxplot(x='Churn',y='tenure',data = df)
plt.title("Tenure vs Churn")
#plt.show()

sns.boxplot(x='Churn',y='MonthlyCharges',data = df)
plt.title("Monthly Charges vs Churn")
#plt.show()

sns.boxplot(x='Churn',y='TotalCharges',data = df)
plt.title("Total Charges  vs Churn")
#plt.show()



print("x----x-----x-----x-----x----Categorical Columns Exploration----x---x----x-----x----x")
cat_col= df.select_dtypes(include = 'str').columns.tolist()
print(cat_col)


print('\n\n\n')
print(df['Contract'].value_counts())
print((pd.crosstab(df['Contract'] , df['Churn'] )))
print('\n\n\n')
print((pd.crosstab(df['InternetService'] , df['Churn'] )))
print('\n\n\n')
print((pd.crosstab(df['PaymentMethod'] , df['Churn'] )))
print('\n\n\n')
print((pd.crosstab(df['PaperlessBilling'] , df['Churn'] )))

sns.countplot(x='Contract' , hue = 'Churn' , data = df)
plt.title("Contract Type vs Churn")
#plt.show()

df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
print(df[['tenure','MonthlyCharges','TotalCharges' ,'Churn_num']])
corr = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_num']].corr()
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
#plt.show()

df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID',axis = 1, inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

print((df.columns))
df_encoded.to_csv('./data_set/TelCom_churn_preprocessed.csv', index=False)

n_df = pd.read_csv('./data_set/TelCom_churn_preprocessed.csv')
print('Column Names : ' ,n_df.columns ,'\n')
print('Null values in the Column :' ,df.isna().sum())


print(df.select_dtypes(include = 'str').columns.tolist())