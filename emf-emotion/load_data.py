import pyedflib
import numpy as np
import os
import glob

class Subject_Session_Data:
        def __init__(self, f, file_count):
                self.file_count = file_count
                self.num_signals = f.signals_in_file
                self.file_duration = f.file_duration
                self.signal_labels = f.getSignalLabels()
                self.data_dictionary = {}

                for i in range(f.signals_in_file):
                        data = f.readSignal(i)
                        self.data_dictionary[self.signal_labels[i]] = data

        def display_info(self):
                print("file number:", self.file_count)
                print("File duration in seconds:", self.file_duration)
                print("Number of signals:", self.num_signals)
                for label in self.signal_labels:
                        print("Label:", label)
                        print(self.data_dictionary[label])

work_dir = os.path.dirname(os.path.dirname(os.getcwd()))
dataset_dir = os.path.join(work_dir, "enterface06_EMOBRAIN", "Data")
dataset_common_dir = os.path.join(dataset_dir, "Common")
dataset_EEG_dir = os.path.join(dataset_dir, "EEG")
dataset_fNIRS_dir = os.path.join(dataset_dir, "fNIRS")

print("working with dataset directory:", dataset_dir)

# Construct the search pattern
pattern = os.path.join(dataset_EEG_dir, "*.bdf")

# Find all files in the directory matching the pattern
bdf_files = glob.glob(pattern)

count = 1
data_list = []
for file in bdf_files:
        # Open the BDF file
        try:
                with pyedflib.EdfReader(file) as f:
                        print("reading file :", count)
                        data_list.append(Subject_Session_Data(f, count))
                        '''
                        # Get general information
                        print("reading file :", count)
                        print("File duration in seconds:", f.file_duration)
                        print("Number of signals:", f.signals_in_file)
                        signal_labels = f.getSignalLabels()
                        print("Signal labels:", signal_labels)

                        # Read data from each signal
                        for i in range(f.signals_in_file):
                                data = f.readSignal(i)
                                print(f"Data from signal {signal_labels[i]}:", data)
                        '''
        except (OSError):
                print("reading file :", count, "FAIL")
        count += 1

data_list[0].display_info()