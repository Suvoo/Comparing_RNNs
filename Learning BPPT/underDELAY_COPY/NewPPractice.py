from cgi import print_directory


def mul(i):
    return i * i

# Using the map function
num = [3, 5, 7, 11, 13]
x = map(mul, num)

print (x)

# print(list(x))
for i in x:
    print(i)

a = ["John", "Charles", "Mike"]
b = ["Jenny", "Christy", "Monica", "Vicky"]

steps = 4
ans = zip(range(steps),a, b)
print(tuple(ans))