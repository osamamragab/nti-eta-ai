import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, svm, inspection, model_selection, metrics

iris = datasets.load_iris()
# print(iris)
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
df["target"] = iris.target
print(df)

x = df.drop(["petal length (cm)", "petal width (cm)", "target"], axis="columns")
y = df.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=1
)
print(len(x_train))

model = svm.SVC(kernel="linear", random_state=32)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(accuracy)

title = "SVC with linear kernel"

x0 = x_train["sepal length (cm)"]
x1 = x_train["sepal width (cm)"]

fig, ax = plt.subplots(figsize=(8, 6))
disp = inspection.DecisionBoundaryDisplay.from_estimator(
    model,
    x_train,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.75,
    ax=ax,
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
)
ax.scatter(x0, x1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()

precision = metrics.precision_score(y_test, y_pred, average="weighted") * 100
recall = metrics.recall_score(y_test, y_pred, average="weighted") * 100
f1 = metrics.f1_score(y_test, y_pred, average="weighted") * 100

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Compute confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# Generate classification report
report = metrics.classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(accuracy)


dataset = pd.read_csv("Position_Salaries.csv")
print(dataset)

xl = dataset.iloc[:, 1:-1].values
yp = dataset.iloc[:, -1].values
print(xl)
print(yp)

yp = yp.reshape(-1, 1)
print(yp)

# stds_x = scal
