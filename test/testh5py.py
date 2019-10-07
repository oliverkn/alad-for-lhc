import numpy as np
import h5py

f = h5py.File('/home/oliverkn/pro/6021/data_hlf.hdf5', "r")
# %%
print(list(f.keys()))

d = f['data']
print(d.shape)

a = d[:100]
a = np.array(a)
print(a)
print(type(a))

for i in range(0):
    print(i)

# %%
d = {}
d['n'] = 1
d['n'] = d['n'] + 1

print(d)
