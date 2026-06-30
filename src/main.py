import pandas as pd
import numpy  as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("fake_reviews_dataset.csv")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df["label"].value_counts())

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

print("\nShape after TF-IDF:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Reviews:", X_train.shape[0])
print("Testing Reviews:", X_test.shape[0])

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nAI training completed successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))