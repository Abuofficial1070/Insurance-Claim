import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("C:/Users/abdul/Downloads/insurance_claims.csv")

print("Dataset Loaded Successfully\n")

print(df.head())



X = df.drop("fraud", axis=1)

y = df["fraud"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))


joblib.dump(model, "fraud_model.pkl")

joblib.dump(X.columns.tolist(), "features.pkl")

print("\nModel Saved Successfully")

