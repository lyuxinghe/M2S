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