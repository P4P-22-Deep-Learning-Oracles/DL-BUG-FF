import numpy as np 
import matplotlib.pyplot as plt 

A = np.matrix([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
b = np.matrix([20, 33, 36])
result = np.linalg.solve(A, b, T)
print(result)