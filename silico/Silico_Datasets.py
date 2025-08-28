
PAST_LEN   = 288
FUTURE_LEN = 144
import numpy as np
X     = np.load(r'silico_data/X_UV.npy')
Y     = np.load(r'silico_data/Y_UV.npy')
X_mu  = np.load(r'silico_data/X_mean.npy')
X_std = np.load(r'silico_data/X_std.npy')
Y_mu  = np.load(r'silico_data/Y_mean.npy')
Y_std = np.load(r'silico_data/Y_std.npy')
