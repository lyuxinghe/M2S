import numpy as np
from scipy import signal
import math

from preprocess import butter_bandpass_filter

class EEGFeatureExtraction:
    
    def __init__(self, data, fs=1024):
        self.data = data
        self.fs = fs
        self.num_samples = len(data)
        self.num_channels = data[1].num_channels
        self.num_truncations = len(data[1].block_sample_ranges)
        self.target_channels = ['F3', 'F4', 'T7', 'Fp1', 'Fp2', 'T8', 'F7', 'F8', 'O1', 'P7', 'P8', 'O2']
        print()

    def display_info(self):
            # samples = {}
            print("Number of samples:", self.num_samples)
            print("Number of channels:", self.num_channels)
            print("Number of truncations:", self.num_truncations)
            print("Target channels:", self.target_channels)

    def extract_features(self):
        results = np.zeros(shape = (self.num_samples, len(self.target_channels) * self.num_truncations))
        for sample_id, sample in enumerate(self.data):
            for channel_id, channel in enumerate(self.target_channels):
                assert channel in sample.signal_labels, channel + " target channel is not present in the sample data!"
                channel_signal = sample.data_dictionary[channel]
                for band_id, truncation_range in enumerate(sample.block_sample_ranges):
                    truncation = channel_signal[truncation_range[0]:truncation_range[1]]
                    filtered = butter_bandpass_filter(truncation, 4, 45, self.fs)
                    f, Pxx_den = signal.welch(truncation, self.fs, nperseg=256, noverlap=128)
                    #print Pxx_den.shape, Pxx_den
                
                    results[sample_id][channel_id*self.num_truncations + band_id] = math.log(np.max(Pxx_den))

        return results
    