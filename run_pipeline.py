import pandas as pd
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = load_iris()

X = pd.DataFrame(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train,y_train)
prediction = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Accuracy:",accuracy)
joblib.dump(
    pipeline,
    "artifacts/model.pkl"
)
metrics = {
    "accuracy": accuracy
}


with open(
    "artifacts/metrics.json",
    "w"
) as file:
    json.dump(metrics,file)