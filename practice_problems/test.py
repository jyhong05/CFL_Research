import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

test1 = np.array([[1,2,3],
                  [4,5,6]])
test2 = np.array([[2,3,4],
                  [5,6,7]])
test3 = np.array([[3,4,5],
                  [6,7,8]])

print(np.mean([test1, test2, test3], axis=1))