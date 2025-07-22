import pandas as pd
import numpy as np
from sklearn import preprocessing, impute

np.set_printoptions(precision=1)

names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = pd.read_csv("pima-indians-diabetes.csv")
array = dataframe.values


print("Min Max")
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform(array)
print(data_rescaled[0:5, :])


print("Norm 1")
data_scaler = preprocessing.Normalizer(norm="l1")
data_rescaled = data_scaler.fit(array).transform(array)
print(data_rescaled)


print("Norm 2")
data_scaler = preprocessing.Normalizer(norm="l2")
data_rescaled = data_scaler.fit(array).transform(array)
print(data_rescaled)


print("StandardScaler")
data_scaler = preprocessing.StandardScaler()
data_rescaled = data_scaler.fit(array).transform(array)
print(data_rescaled)


print("Binarizer")
binaryrizer = preprocessing.Binarizer(threshold=0.5)
data_binarized = binaryrizer.fit(array).transform(array)
print(data_binarized)


print("encoding")
input_labels = ["red", "black", "red", "green", "black", "yellow", "white"]
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
encoded_values = encoder.transform(input_labels)
print("labels:", input_labels)
print("encoded:", encoded_values)

decoded_list = encoder.inverse_transform(encoded_values)
print("encoded:", encoded_values)
print("decoded:", decoded_list)


print("imputer")
data = np.array(
    [
        [1, 2, np.nan],
        [3, np.nan, 1],
        [5, np.nan, 0],
        [np.nan, 4, 6],
        [5, 0, np.nan],
        [4, 5, 5],
    ]
)
print(data)

imp = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
fitted = imp.fit(data).transform(data)
print(fitted)
