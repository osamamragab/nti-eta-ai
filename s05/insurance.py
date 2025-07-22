import pandas as pd
from sklearn import preprocessing, linear_model, model_selection, metrics

df = pd.read_csv("insurance.csv")
print(df.head())
print("shape:", df.shape)

print("missing values:")
print(df.isnull().sum())

df_input = df.drop(columns="expenses")
df_target = df.expenses
print(df_input)
print(df_target)

label_encoder = preprocessing.LabelEncoder()
input_columns = ["sex", "smoker", "region"]

for col in input_columns:
    df_input[col] = label_encoder.fit_transform(df_input[col])

print(df_input)

scale_column = "expenses"
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(df[[scale_column]])
print(scaled_data)

x_train, x_test, y_train, y_test = model_selection.train_test_split(df_input, df_target, test_size=0.2, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

prediction_test = reg.predict(x_test)
print(prediction_test)

mae_value= metrics.mean_absolute_error(y_test, prediction_test, multioutput="uniform_average")
print("Mean Absolute Error:", mae_value)

mse_value = metrics.mean_squared_error(y_test, prediction_test, multioutput="uniform_average")
print("Mean Sequared Error:", mse_value)
