import pandas as pd

# import numpy as np
from sklearn import preprocessing

data = pd.read_csv("titanic.csv")
encoder = preprocessing.LabelEncoder()
labels = ["Gender", "Embarked" ]
df = pd.DataFrame()

for label in labels:
    df[label] = encoder.fit_transform(data[label])

print(df)
