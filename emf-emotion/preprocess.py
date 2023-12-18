import numpy as np

from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import FastICA
import matplotlib as plt

# Bandpass filter
# For: Environmental noise removal by applying a bandpass filter in the 4-45 Hz range
# fs: sample rate, lowcut/ highcut: band cutoffs
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    #sos = butter(order, [low, high], analog=False, fs=fs, btype='band', output='sos')
    sos = butter(order, [lowcut, highcut], analog=False, fs=fs, btype='band', output='sos')
    filtered = sosfiltfilt(sos, data)
    return filtered

# input: array of size [num_channels, channel_signals]
# output: fitted_ica, components
# reference for artifacts pattern: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6427454/ , https://www.learningeeg.com/artifacts
def show_components(raw_eeg_data, fs=1024, low_cut=4, high_cut=45, n_components=4, bp=True):
    # bandpass
    if bp:
        filtered = np.zeros_like(raw_eeg_data)
        for index, _ in enumerate(raw_eeg_data):
            filtered[index] = butter_bandpass_filter(raw_eeg_data[index], low_cut, high_cut, fs)
        filtered = filtered.T
    else:
        filtered = raw_eeg_data.T

    # ica
    fitted_ica = FastICA(n_components)
    fitted_ica.fit(filtered)
    components = fitted_ica.transform(filtered)

    for i in range(n_components):
        plt.figure(i)
        plt.title('ICA component ' + str(i))
        plt.plot(components[:, i])
    plt.show()

    return fitted_ica, components

# input: list of components index to be removed, fitted_ica, components
# output: array of size [num_channels, channel_signals]
def clean_components(removal, fitted_ica, components):
    # remove artifacts
    for i in removal:
        components[:, i] = 0
    # restore signals
    clean_data = fitted_ica.inverse_transform(components).T

    return clean_data