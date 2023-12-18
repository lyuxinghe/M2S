import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, dropout=0.4, num_layers=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
