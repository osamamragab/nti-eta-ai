import numpy as np


vec1d_row = np.array([1, 2, 3])
vec1d_col = np.array([[1], [2], [3]])
print("row vec1d:", vec1d_row)
print("col vec1d:", vec1d_col)

vec2d = np.array([[1,2,3], [4,5,6]])
print("vec2d:", vec2d)


list1 = np.arange(start=1, stop=13, step=1)
print("list1:", list1)

arr = np.array(list1.reshape(3,4))
print("arr:", arr)

ls1= [1,2,3]
ls2 = [4,5,6]

arr1 = np.array(ls1)
arr2 = np.array(ls2)

arr_sum = arr1+arr2
print(arr_sum)
