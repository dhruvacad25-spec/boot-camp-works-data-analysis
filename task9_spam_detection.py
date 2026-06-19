

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv "

#Load the Dataset
df=pd.read_csv(url)
print(df.head())
print(df.info())

#Prepare features(x) and target (y)
#1. Clean total charges
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges']= df['TotalCharges'].fillna(0)
#2. Drop identifier column
df=df.drop(columns=['customerID'])
#3. Separate target from features
y=df['Churn']
x=df.drop(columns=['Churn'])
#4. Encode binary Yes/No columns as 0/1
binary_columns=['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_columns:
  x[col]=x[col].replace({'Yes':1, 'No':0})
x['gender']=x['gender'].replace({'Female':1, 'Male':0})
multiclass_columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
x=pd.get_dummies(x, columns=multiclass_columns, drop_first=True)
print("\nShape after encoding: ", x.shape)
print("Columns: ", list(x.columns))

# Encode target labels to numerical values
label_encoder = LabelEncoder()
y= label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#Scaling only numeric columns
numeric_cols=['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train[numeric_cols])
x_test = scaler.transform(x_test[numeric_cols])

metrics = {
    'Eucleidian': ('minkowski',2),
    'Manhattan': ('minkowski',1),
    'Minkowski(p=3)': ('minkowski',3),
}

for name ,(metric_name,p) in metrics.items():
  print(f"\n=== {name}  ===")

  # Confusion Matrix for K=5 (as originally defined)
  knn = KNeighborsClassifier(n_neighbors=5, metric=metric_name, p=p)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  print(f"Accuracy (K=5): {acc}")

  cm=confusion_matrix(y_test,y_pred)
  plt.figure(figsize=(6, 5))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title(f'Confusion Matrix for KNN- {name} (K=5)')
  plt.show()

  # Accuracy vs K plot
  k_values = list(range(1, 21))
  accuracy_scores = []

  for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k, metric=metric_name, p=p)
    knn_k.fit(x_train, y_train)
    y_pred_k = knn_k.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_k))

  # Find the best K for the current metric
  best_k_index = np.argmax(accuracy_scores)
  best_k = k_values[best_k_index]
  best_accuracy = accuracy_scores[best_k_index]
  print(f"\nFor {name} Distance, the best K is {best_k} with an accuracy of {best_accuracy:.4f}\n")

  plt.figure(figsize=(10, 6))
  plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
  plt.title(f'Accuracy vs. K for KNN with {name} Distance')
  plt.xlabel('Number of Neighbors (K)')
  plt.ylabel('Accuracy')
  plt.xticks(k_values) # Ensure all K values are shown on x-axis
  plt.grid(True)
  plt.show()
