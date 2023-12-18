import pyedflib
import numpy as np
import os
import glob
import pickle

class Subject_Session_Data:
        def __init__(self, f, file_count, participant_number, session_number, marks, block_sample_ranges):
                self.file_count = file_count
                self.num_channels = f.signals_in_file
                self.file_duration = f.file_duration
                self.signal_labels = f.getSignalLabels()
                self.participant_number = participant_number
                self.session_number = session_number
                self.data_dictionary = {}
                self.marks = marks
                self.emotions = []
                self.block_sample_ranges = block_sample_ranges

                for i in range(f.signals_in_file):
                        data = f.readSignal(i)
                        self.data_dictionary[self.signal_labels[i]] = data

        def display_info(self):
                # samples = {}
                print("file number:", self.file_count)
                print("File duration in seconds:", self.file_duration)
                print("Number of channels:", self.num_channels)
                print("Number of samples:", len(self.data_dictionary[self.signal_labels[0]]))
                print("Participant number:", self.participant_number)
                print("Session number:", self.session_number)
                print("Emotions:", self.emotions)
                print("Signal labels:", self.signal_labels)

                #################### TO VIEW SELECT CHANNEL SIGNAL PLOTS #############################
                # for label in self.signal_labels:
                        # print("Label:", label)
                        # print(self.data_dictionary[label])
                # figure, axis = plt.subplots(3, 3) 
                # for i in range(3):
                #         for j in range(3):
                #                 axis[i,j].plot([x for x in range(len(self.data_dictionary[self.signal_labels[3*i+j]]))], self.data_dictionary[self.signal_labels[3*i+j]]) 
                #                 axis[i,j].set_title("First_signal_"+self.signal_labels[3*i + j]) 



def load_raw_all():
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
                                # marks = []
                                print("reading file :", count)
                                participant_index = file.find("Part")
                                session_index = file.find("SES")
                                participant_number = int(file[participant_index+4])
                                # print("Participant number:", participant_number)
                                session_number = int(file[session_index+3])
                                # print("Session number:", session_number)
                                if participant_number == 2 and session_number == 1:
                                        continue
                                fi = open(file+".mrk", "r")
                                fi.readline()
                                inferred_temp_marks = []
                                for line in fi:
                                        temp_marks = line.split('\t')[1:]
                                        temp_marks[-1] = temp_marks[-1][:-1]
                                        if temp_marks[-1] == '"255"':
                                                if temp_marks[0] == temp_marks[1]:
                                                        inferred_temp_marks.append(int(temp_marks[0]))
                                                else:
                                                        print("Irregular:", temp_marks)

                                block_sample_ranges = []
                                for trigger in inferred_temp_marks:
                                        if participant_number == 1 and session_number == 1:
                                                start_index = trigger + 768
                                                end_index = start_index + 3200
                                                block_sample_ranges.append((start_index,end_index))
                                        else:
                                                start_index = trigger + 3072
                                                end_index = start_index + 12800
                                                block_sample_ranges.append((start_index,end_index))

                                data_list.append(Subject_Session_Data(f, count, participant_number, session_number, inferred_temp_marks, block_sample_ranges))
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

        emotion_classes = os.path.join(dataset_common_dir, "IAPS_Classes_EEG_fNIRS.txt")
        emotion_file = open(emotion_classes, "r")
        session1 = []
        session2 = []
        session3 = []
        for line in emotion_file:
                temp_emotions = line.split("\t")
                temp_emotions[-1] = temp_emotions[-1][:-1]
                session1.append(temp_emotions[0])
                session2.append(temp_emotions[1])
                session3.append(temp_emotions[2])

        for subject_session in data_list:
                if subject_session.session_number == 1:
                        subject_session.emotions = session1
                elif subject_session.session_number == 2:
                        subject_session.emotions = session2
                elif subject_session.session_number == 3:
                        subject_session.emotions = session3

        ################################ VALIDATION OF .MRK FILE DO NOT DELETE ##############################
        # seconds = {}
        # j_index = 0
        # for j in range(len(data_list)):
        #         flag = 0
        #         print("Number of marks in session:", len(data_list[j].marks))
        #         print("Number of emotions:", len(data_list[j].emotions))
        #         print(data_list[j].marks)
        #         print("number of signals:", len(data_list[j].data_dictionary[data_list[j].signal_labels[0]]))
        #         print("participant number:", data_list[j].participant_number)
        #         print("session number:", data_list[j].session_number)
                
        #         for i in range(len(data_list[j].marks)):
        #                 if i == 0:
        #                         continue
        #                 else:
        #                         if data_list[j].participant_number == 1 and data_list[j].session_number == 1:
        #                                 temp_seconds = (data_list[j].marks[i] - data_list[j].marks[i-1])/256
        #                                 flag = 1
        #                         else:
        #                                 temp_seconds = (data_list[j].marks[i] - data_list[j].marks[i-1])/1024
        #                         if flag == 1:
        #                                 j_index = j

        #                         if j in seconds.keys():
        #                                 seconds[j].append(temp_seconds)
        #                         else:
        #                                 seconds[j] = [temp_seconds]

        # for key in seconds.keys():
        #         if key == j_index:
        #                 print("First participant, first session")
        #                 print("Minimum time for one trial:", min(seconds[key]))
        #                 print("Maximum time for one trial:", max(seconds[key]))
        #                 print("Mean time for one trial:", sum(seconds[key])/len(seconds[key]))
        #         else:
        #                 print("Minimum time for one trial:", min(seconds[key]))
        #                 print("Maximum time for one trial:", max(seconds[key]))
        #                 print("Mean time for one trial:", sum(seconds[key])/len(seconds[key]))

        return data_list

def load_raw_single(file_index):
        pkl_file_name = 'eeg_' + str(file_index) + '.pkl' 
        with open(pkl_file_name, 'rb') as file:
                loaded_data = pickle.load(file)
        return loaded_data

def load_clean_single(file_index):
        pkl_file_name = 'clean_' + str(file_index) + '.pkl' 
        with open(pkl_file_name, 'rb') as file:
                loaded_data = pickle.load(file)
        return loaded_data

