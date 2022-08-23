from enum import Enum
# https://docs.python.org/3/library/enum.html

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.RED)
print(repr(Color.RED))

for col in Color:
    print(col)

print(Color['RED'])

print(Color.RED.name)
print(Color.RED.value)