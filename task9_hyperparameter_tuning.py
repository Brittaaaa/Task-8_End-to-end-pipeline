import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
data = pd.read_csv("data/iris.csv")

print(data.head())
X = data.drop("target", axis=1)

y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Baseline Model

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    prediction
)

print("Baseline Accuracy:", baseline_accuracy)
# Hyperparameter tuning

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10]
}


grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=2,
    scoring="accuracy"
)


grid.fit(X_train, y_train)
print("Best Parameters:")
print(grid.best_params_)
print("Best CV Score:")
print(grid.best_score_)
# Tuned model evaluation

best_model = grid.best_estimator_

tuned_prediction = best_model.predict(X_test)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_prediction
)

print("Tuned Accuracy:")
print(tuned_accuracy)