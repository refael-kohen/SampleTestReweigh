import numpy as np

dim = 15


sample = np.random.normal(loc=0, scale=1, size=(500000, dim)) ** 4
values_under_thresh = []
for _ in range(100):
    for L in np.arange(100):
        frac_above_L = np.mean(np.sqrt(np.sum(sample, axis=1)) > L)
        if frac_above_L < 0.01:
            values_under_thresh.append(L)
            break
print('mean L is {}'.format(np.mean(np.asarray(values_under_thresh, dtype=np.double))))
