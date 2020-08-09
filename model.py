import torch.nn as nn
import torch.nn.functional as F
from torch import cat, transpose


class ResCNN_block(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, kernel_size=3):
        super(ResCNN_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.dropout1 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=hidden_channels)

        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.dropout2 = nn.Dropout(0.1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=hidden_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.batch_norm1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.batch_norm2(x)
        x = F.gelu(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, hidden_channels=32, kernel_size=3):
        super(ResCNN, self).__init__()
        self.block1 = ResCNN_block(32, hidden_channels=hidden_channels, kernel_size=kernel_size)
        self.block2 = ResCNN_block(64, hidden_channels=hidden_channels, kernel_size=kernel_size)
        self.block3 = ResCNN_block(96, hidden_channels=hidden_channels, kernel_size=kernel_size)

    def forward(self, x):  # shape (batch_size, channels=1, frequency=128, time)
        out1 = self.block1(x)
        out2 = self.block2(cat((x, out1), dim=1))
        return self.block3(cat((x, out1, out2), dim=1))


class BiRNN_block(nn.Module):
    def __init__(self, input_size, output_size):
        super(BiRNN_block, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=output_size//2, num_layers=1, batch_first=True,
                            bidirectional=True)  # //2 because bidirectional doubles the output size (forward-backward)s
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x desired shape (time, batch_size, input_size)
        x = F.gelu(x)
        x, _ = self.LSTM(x)
        x = self.dropout(x)
        return x


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.LSTM1 = BiRNN_block(input_size=input_size, output_size=hidden_size)
        self.LSTM2 = BiRNN_block(input_size=hidden_size, output_size=hidden_size)
        self.LSTM3 = BiRNN_block(input_size=hidden_size, output_size=output_size)

    def forward(self, x):
        x = self.LSTM1(x)  # shape (time, batch_size, hidden_size)
        x = self.LSTM2(x)
        return self.LSTM3(x)


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=29):
        super(Classifier, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class ParrotModel(nn.Module):
    def __init__(self):
        super(ParrotModel, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=3 // 2)
        self.encoder = ResCNN()
        self.FC = nn.Linear(in_features=32*128, out_features=512)  # last conv to rnn in
        self.RNN = BiRNN(input_size=512, hidden_size=512, output_size=512)
        self.classifier = Classifier(512, 256, num_classes=29)  # 29 characters

    def forward(self, x):       # shape (batch_size, frequencies=128, time)
        x = x.unsqueeze(1)      # shape (batch_size, channels=1, frequencies=128, time)
        x = self.conv0(x)       # shape (batch_size, channels=32, frequencies=128, time)
        x = self.encoder(x)     # shape (batch_size, channels=32, frequencies=128, time)
        #            unrolling to shape (batch_size, time, 128*32)
        x = transpose(x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]), 1, 2)
        x = self.FC(x)          # shape (batch_size, time, 512)
        x = transpose(x, 1, 0)  # shape (time, batch_size, 512)
        x = self.RNN(x)         # shape (time, batch_size, 512)
        x = transpose(x, 1, 0)  # shape (batch_size, time, 512)
        return self.classifier(x)  # shape (batch_size, time, 29)
