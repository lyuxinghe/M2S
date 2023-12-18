import numpy as np
from scipy import signal
import math
from scipy.fft import fft


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



def get_features(raw_data_list, clean_data_list, selected_channels):
    features = []
    for cur_dataset, cur_clean in zip(raw_data_list, clean_data_list):
        dataset_features = []
        for channel in selected_channels:
            channel_features = []
            assert channel in cur_dataset.signal_labels, channel + " target channel is not present in the sample data!"
            #channel_signal = cur_dataset.data_dictionary[channel]
            for band_id, truncation_range in enumerate(cur_dataset.block_sample_ranges):
                cur_channel_index = cur_dataset.signal_labels.index(channel)
                truncation = cur_clean[cur_channel_index][truncation_range[0]:truncation_range[1]]
                channel_features.append(fft(truncation))
            dataset_features.append(channel_features)
        features.append(dataset_features)

    features = np.hstack(np.hstack(features))
    return features