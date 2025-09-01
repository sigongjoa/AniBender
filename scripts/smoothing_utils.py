import numpy as np
from scipy.signal import savgol_filter

class OneEuroFilter:
    def __init__(self, freq=30, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.last_value = None
        self.last_derivative = None

    def __call__(self, x):
        if self.last_value is None:
            self.last_value = x
            self.last_derivative = np.zeros_like(x)
            return x
        # 1€ 필터 기본 구현 (간단화)
        alpha = self.smoothing_factor(self.freq, self.min_cutoff)
        dx = (x - self.last_value) * self.freq
        self.last_derivative = self.exponential_smoothing(alpha, dx, self.last_derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self.last_derivative)
        alpha = self.smoothing_factor(self.freq, cutoff)
        self.last_value = self.exponential_smoothing(alpha, x, self.last_value)
        return self.last_value

    def smoothing_factor(self, freq, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def exponential_smoothing(self, alpha, x, prev):
        return alpha * x + (1 - alpha) * prev

def moving_average_filter(data, window_size=5):
    # data: (num_frames, num_joints, num_coords)
    # Apply moving average along the frame axis (axis=0)
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[1]): # Iterate over joints
        for j in range(data.shape[2]): # Iterate over coordinates (x, y, z, visibility)
            smoothed_data[:, i, j] = np.convolve(data[:, i, j], np.ones(window_size)/window_size, mode="same")
    return smoothed_data

def savgol_filtering(data, window_length=11, polyorder=2):
    # data: (num_frames, num_joints, num_coords)
    # Apply Savitzky-Golay filter along the frame axis (axis=0)
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[1]): # Iterate over joints
        for j in range(data.shape[2]): # Iterate over coordinates (x, y, z, visibility)
            smoothed_data[:, i, j] = savgol_filter(data[:, i, j], window_length, polyorder)
    return smoothed_data
