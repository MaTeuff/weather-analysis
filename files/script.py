import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data


timestamps, measurements, datetimes = load_data()

weights_time = np.ones_like(timestamps)
for i in range(len(weights_time)):
    weights_time[i] = 1. / (np.sum(np.abs(timestamps - timestamps[i]) < 60 * 12))
weights_time /= weights_time.mean()

measurements_mean = measurements.mean()
measurements_std = measurements.std()
measurements_normalized = (measurements - measurements_mean) / measurements_std

timestamps_mean = timestamps.mean()
timestamps_std = timestamps.std()
timestamps_normalized = (timestamps - timestamps_mean) / timestamps_std


# y_pred_final = y_pred.detach().numpy() * measurements_std + measurements_mean
# y_final = y.detach().numpy() * measurements_std + measurements_mean
# plt.figure(), plt.plot(timestamps_normalized, y_pred_final), plt.plot(timestamps_normalized, y_final), plt.show()

