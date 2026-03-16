import numpy as np
import common

from dezero import Variable
import dezero.functions as F


if __name__ =="__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    a, b = Variable(a), Variable(b)
    c = F.matmul(a, b)
    print(c)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = F.matmul(a, b)
    print(c)
    