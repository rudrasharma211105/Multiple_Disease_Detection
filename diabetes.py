import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df1 = pd.read_csv('data/diabetes.csv')

# Exploratory Data Analysis
print(df1.isnull().sum())
df1.info()
print(df1.describe())
print(df1.head(6))
print(df1['Pregnancies'].unique())
# sns.pairplot(data=df1)

# Prepare data
x = df1.drop("Outcome",axis=1)
y = df1["Outcome"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=42)

# Train model
lr = LogisticRegression()
lr.fit(x_train,y_train)

# Evaluate model
y_pred = lr.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(lr, "models/model_diabetes.pkl")
