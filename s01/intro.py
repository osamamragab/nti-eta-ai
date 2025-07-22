# printing

print("hola")

"""
this is a doc string/comment
"""
print("bye!")


# vars

name = "John Cena"
_name = "Astroid Destroyer"
name69 = "yomama"
print(f"My name is {name}")
print(f"My name is {_name}")
print(f"My name is {name69}")


# input

# x = int(input("x = "))
# y = int(input("y = "))
# print(f"x + y = {x+y}")


# more vars
a, b, c = 1, 2, 3
d = e = f = 5
print(f"a={a}, b={b}, c={c}")
print(f"d={d}, e={e}, f={f}")


# funcs

def hi():
    print("hey bud")

hi()


# random module

import random

print(random.randrange(10))


# lists

ls = ["l", "o", "l", 6, 9]
print(ls)
print(len(ls))
print(ls[2])
ls[2] = "m"
print(ls)
print(ls[:3])
print(ls[3:])
print(ls[1:4])
print(ls[-4:-1])

if 6 in ls:
    print("we have 6 in there")

ls[0], ls[1] = ["y", "o"]
print(ls)

ls.insert(0, 69)
print(ls)

ls.append(420)
print(ls)

ls.extend([4,2,0])
print(ls)

ls.remove("y")
print(ls)

ls.pop(2)
print(ls)

ls.pop()
print(ls)

del ls[2]
print(ls)

ls.clear()
print(ls)

del ls
# print(ls) # error

ls = [1,2,3,4,5]
for x in ls:
    print(x)

ls = [1,2,3,4,5]
for i in range(len(ls)):
    print(i, ls[i])

ls.sort()
print(ls)

ls.sort(reverse=True)
print(ls)

ls.reverse()
print(ls)

cp = ls.copy()
print(ls)

for i in range(len(cp)):
    cp[i]+=ls[-1]

ls = ls+cp
print(ls)

tup = (1,2,3,4)
print(tup)

tup = list(tup)
tup[3] = 5
tup = tuple(tup)
print(tup)

seto = {1,2,3,4}
seto.add(1)
print(seto)
