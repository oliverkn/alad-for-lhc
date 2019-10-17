import numpy as np

cat = [-1, 0]

min = np.amin(cat)
max = np.amax(cat)

a = np.array([-1, 1, 1, -1, 1])
a = np.clip(a, min, max)
print(a)
