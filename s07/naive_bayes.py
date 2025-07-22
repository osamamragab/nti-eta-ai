import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())

df.drop(
    ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    axis="columns",
    inplace=True,
)
print(df.head())
