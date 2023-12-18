import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from load_data import load_raw_all, load_raw_single, load_clean_single
from preprocess import clean_components, show_components
from feature_extraction import get_features
from networks import LSTMModel

# util function
def extend_lists(*lists):
    if len(lists) == 1:
        return lists[0]
    else:
        merged = []
        for per_list in lists:
            merged.extend(per_list)
        return merged

# preprocess data if needed
def clean_data(file_index):
    # load raw data of specific file index
    raw_data = load_raw_single(file_index)
    eeg_stack = np.array([raw_data.data_dictionary[label] for label in raw_data.signal_labels])
    
    # show ica components
    fitted_ica, components_ica = show_components(eeg_stack)

    # input noise ica component indices to be removed
    input_components = input("Enter components number(s) to be removed, separated by spaces: ")
    string_elements = input_components.split()
    noise_components = [int(element) for element in string_elements]
    print("Your array of integers is:", noise_components)

    # remove noise components and reconstruct clean signal
    restored_data = clean_components(noise_components, fitted_ica, components_ica)

    # ask whetheer to store clean data to disk
    store = input("Enter Y to store the restored data to disk, or otherwise skip: ")
    if store == 'Y' or 'y':
        pkl_file_name = 'clean_' + str(file_index) + '.pkl' 
        with open(pkl_file_name, 'wb') as file:
            pickle.dump(restored_data, file)
        print("Stored")
    else:
        print("Skipped")

    return restored_data



def main():
    file_indices = [1,4,7]
    select1_dataset = load_raw_single(1)
    select2_dataset = load_raw_single(4)
    select3_dataset = load_raw_single(7)

    select1_clean_data = load_raw_single(1)
    select2_clean_data = load_raw_single(4)
    select3_clean_data = load_raw_single(7)

    selected_channels  = ['F3', 'F4']

    emotion_labels = select1_dataset.emotions + select2_dataset.emotions + select3_dataset.emotions
    print("emotion lables:", len(emotion_labels))

    dataset_list = [select1_dataset, select2_dataset, select3_dataset]
    clean_list = [select1_clean_data, select2_clean_data, select3_clean_data]
    features = get_features(dataset_list, clean_list, selected_channels)
    print("features", features.shape)

    # build train/validate and test dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    scaler = StandardScaler()
    df_features = pd.DataFrame(scaler.fit_transform(np.abs(features)))
    enc = LabelEncoder()
    df_labels = pd.DataFrame(enc.fit_transform(emotion_labels))
    X_train,X_test, y_train, y_test = train_test_split(df_features, df_labels, train_size = 0.8, random_state=1)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train.values).float().to(device)
    y_train_tensor = torch.tensor(y_train.values).long().squeeze().to(device)
    X_test_tensor = torch.tensor(X_test.values).float().to(device)
    y_test_tensor = torch.tensor(y_test.values).long().squeeze().to(device)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Model, loss function, and optimizer
    model = LSTMModel(input_dim=X_train.shape[1], hidden_size=512).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop with early stopping
    train_acc, val_acc = [], []
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 6
    trigger_times = 0

    for epoch in tqdm(range(1500)):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_acc.append(100 * correct / total)

        # Early stopping
        if val_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_loss / len(val_loader)
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    # Plotting accuracy and loss
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        model_acc = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Test Accuracy: {model_acc * 100:.3f}%")

        # Confusion Matrix and Classification Report
        y_pred = predicted.cpu().numpy()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title('Confusion Matrix')

        plt.show()