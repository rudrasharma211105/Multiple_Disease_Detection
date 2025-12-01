import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df2 = pd.read_csv('data/cancer.csv')

# Exploratory Data Analysis
print(df2.head(6))
print(df2.isnull().sum())
df2.info()
print(df2.describe())
# sns.pairplot(data=df2)

# Prepare data
x2 = df2.drop("Diagnosis",axis=1)
y2 = df2["Diagnosis"]

# Split data
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2,test_size=0.15,random_state=42)

# Train model
lr2 = LogisticRegression()
lr2.fit(x2_train,y2_train)

# Evaluate model
y2_pred = lr2.predict(x2_test)

print("Accuracy:", accuracy_score(y2_test, y2_pred))
print("Confusion Matrix:\n", confusion_matrix(y2_test, y2_pred))
print("Classification Report:\n", classification_report(y2_test, y2_pred))

# Save model
joblib.dump(lr2, "models/model_cancer.pkl")
