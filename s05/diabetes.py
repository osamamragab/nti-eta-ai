import pandas as pd
from sklearn import linear_model, model_selection, metrics

columns = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]

df = pd.read_csv("diabetes.csv", names=columns)
print(df.head())
print("shape:", df.shape)
print("missing values:")
print(df.isnull().sum())

feature_cols = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
]

x = df[feature_cols]
y = df.label

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, train_size=0.3, random_state=1
)
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

predition_test = model.predict(x_test)
print(predition_test)
print("accuracy:", metrics.accuracy_score(y_test, predition_test))

classification_report = metrics.classification_report(y_test, predition_test)
print("classification report:")
print(classification_report)

confusion_matrix = metrics.confusion_matrix(y_test, predition_test)
print("confusion matrix:")
print(confusion_matrix)
