import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df3 = pd.read_csv('data/heart.csv')

# Exploratory Data Analysis
print(df3.head(6))
print(df3.isnull().sum())
df3.info()
print(df3.describe())
# sns.pairplot(data=df3)

# Prepare data
x3 = df3.drop("condition",axis=1)
y3 = df3["condition"]

# Split data
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3,test_size=0.15,random_state=42)

# Train model
lr3 = LogisticRegression()
lr3.fit(x3_train,y3_train)

# Evaluate model
y3_pred = lr3.predict(x3_test)

print("Accuracy:", accuracy_score(y3_test, y3_pred))
print("Confusion Matrix:\n", confusion_matrix(y3_test, y3_pred))
print("Classification Report:\n", classification_report(y3_test, y3_pred))

# Save model
joblib.dump(lr3, "models/model_heart.pkl")
