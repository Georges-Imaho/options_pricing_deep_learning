T = 1.0
dt=1/252

N_steps = int (T/dt)
n_paths = 1000

import numpy as np

Z = np.random.normal(0, 1, size=(n_paths, N_steps))

print(Z)
print(len(Z))
