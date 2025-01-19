import numpy as np

path = 'Track_1/training_log/long_open_lock_2025-01-16_18-06-16.450/evaluations.npz'

data = np.load(path)
decode_data = {k: v for k, v in data.items()}

print(data.keys())
