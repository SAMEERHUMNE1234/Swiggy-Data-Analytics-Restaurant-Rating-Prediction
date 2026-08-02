import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# load dataset:

df = pd.read_csv("Restaurant.csv")

#check column and shape

print(df.head())
print(df.info())
print(df.shape)

#Step 2: Data Cleaning
#Check if Cuisine and Cuisine.1 are identical

print((df["Cuisine"] == df["Cuisine.1"]).all())
print(df[["Cuisine", "Cuisine.1"]].head(20))

diff = df[df["Cuisine"] != df["Cuisine.1"]]
print("Different rows:", len(diff))
print(diff[["Cuisine", "Cuisine.1"]].head(20))

print(df.drop(columns=["Cuisine.1"], inplace=True))

#Check Missing Values
print(df.isnull().sum())

#Keep Only Restaurants with Ratings
df = df.dropna(subset=["Rating"])
print(df)
print(df.isnull().sum())

# Create the Target Variable
df["High_Rated"] = (df["Rating"] >= 4.0).astype(int)

#Check the class distribution:
print(df["High_Rated"].value_counts())
print(df["Rating_count"].value_counts())
print(df["Rating_count"].unique())

#Convert Rating_count to Numeric
rating_map = {
    "20+ ratings": 20,
    "50+ ratings": 50,
    "100+ ratings": 100,
    "500+ ratings": 500,
    "1K+ ratings": 1000,
    "5K+ ratings": 5000,
    "10K+ ratings": 10000
}

#Remove Unnecessary Columns
df["Rating_count"] = df["Rating_count"].map(rating_map)
print(df.head())
print(df.dtypes)

df = df.drop(columns=["Restaurant_id", "Name", "Rating"])
print(df.columns)

#Encode Categorical Features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Country"] = le.fit_transform(df["Country"])
df["City"] = le.fit_transform(df["City"])
df["Cuisine"] = le.fit_transform(df["Cuisine"])

#Define Features and Target
X = df.drop("High_Rated", axis=1)
y = df["High_Rated"]

#Split the Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Check the shapes:
print(X_train.shape)
print(X_test.shape)

#Train Your First Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)

#Evaluate the Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

#Before improving the dataset, compare multiple models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df)