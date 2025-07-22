import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qtagg")

data = pd.read_csv("data.csv")

print(data.to_string())
print(data.head(10))
print(data.dtypes)
print(data.info())


data = pd.read_csv("test.csv")
print(data.to_string())
print(data.dtypes)
data["rest"] = data["rest"].str.strip("%").astype("float")

print(data["rest"].sum())
print(data["valid"].describe())

plt.hist(data["Rating "])
plt.title("average rating for movies")
#plt.show()

print("drop inplace=False")
copy = data.drop(data[data["Rating "]<5].index, inplace=False)
print(copy)

print("drop inplace=True")
data.drop(data[data["Rating "]<5].index, inplace=True)
print(data)

data.loc[data["Rating "]>5, "Rating "] = 5
print(data)


assert data["Rating "].max() <= 5


data = pd.read_csv("data.csv")
duplicated = data.duplicated()
print(data.to_string())
print(data.head(10))
print(data.dtypes)
print(data.info())
print(duplicated)
print(data[duplicated])

columns = ["Duration", "Pulse", "Maxpulse", "Calories", "Precentage"]
duplicates = data.duplicated(subset=columns, keep=False)
print(duplicates)

duplicates = data.duplicated(subset=columns, keep="first")
print(duplicates)

duplicates = data.duplicated(subset=columns, keep="last")
print(duplicates)

data.drop_duplicates(inplace=True)
print(data)

data = pd.read_excel("presidents_names.xlsx")
print(data.to_string())

print(data.loc[[1,2],["Name", "born"]])


with pd.ExcelFile("presidents_names.xlsx") as xls:
    df1 = pd.read_excel(xls, "Sheet1")
    df2 = pd.read_excel(xls, "Sheet2")
    print(df1[0:3]["Name"])
    print(df2[0:3]["Name"])
