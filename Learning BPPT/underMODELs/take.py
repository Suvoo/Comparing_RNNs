#Importing numpy
import numpy as np

#We will create a 2D array
#Of shape 4x3
# arr = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9), (50, 51, 52)])
# #Printing the array
# print("The array is: ")
# print(arr)

# #Printing values mode=warp
# indices = [0, 6]
# print("Values at position 0 and 6 are [mode=warp]:\n ",
#       np.take(arr, indices, axis=1))
      
x = [100,6,5]
ans = x[1:][::-1]
print(ans)
# #Printing values mode=clip
# indices = [1, 5]
# print("Values at position 1 and 5 are [mode=clip]:\n ",
#       np.take(arr, indices, axis=1, mode='clip'))

# #Printing values mode=raise
# indices = [2, 7]
# print("Values at position 2 and 7 are [mode=raise]:\n ",
#       np.take(arr, indices, axis=1, mode='raise'))