from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class PoseS1(nn.Module):
    def __init__(self):
        super(PoseS1, self).__init__()
        """
        PoseS1
        输入归一化基于根坐标系的加速度和旋转矩阵
        输入除关节点外5个关节点的坐标，以根坐标系为准
        """
        self.L1 = nn.Dropout(p=0.2)
        self.L2 = nn.Linear(72, 256)  # relu
        self.L3 = nn.LSTM(256, 256, bidirectional=True, batch_first=True, num_layers=2)
        #self.L4 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.L5= nn.Linear(512, 15)

    def forward(self, inputs, seq_lens):
        total_length = inputs.shape[1]
        output = self.L1(inputs)
        output = F.relu(self.L2(output))

        x_packed = pack_padded_sequence(output, seq_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.L3(x_packed)
        #output, _ = self.L4(output)
        output, length = pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = self.L5(output)
        return output

class PoseS2(nn.Module):
    def __init__(self):
        super(PoseS2, self).__init__()
        """
        PoseS2
        输入x1
        输入除关节点外23个关节点的坐标，以根坐标系为准
        """
        self.L1 = nn.Dropout(p=0.2)
        self.L2 = nn.Linear(87, 64)  # relu
        self.L3 = nn.LSTM(64, 64, bidirectional=True, batch_first=True, num_layers=2)
        #self.L4 = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        self.L5= nn.Linear(128, 69)

    def forward(self, inputs,  seq_lens):
        total_length = inputs.shape[1]
        output = self.L1(inputs)
        output = F.relu(self.L2(output))

        x_packed = pack_padded_sequence(output, seq_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.L3(x_packed)
        #output, _ = self.L4(output)
        output, length = pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = self.L5(output)
        return output

class PoseS3(nn.Module):
    def __init__(self):
        super(PoseS3, self).__init__()
        """
        PoseS3
        输入x2
        输入除关节点外23个关节点的旋转
        """
        self.L1 = nn.Dropout(p=0.2)
        self.L2 = nn.Linear(141, 128)  # relu
        self.L3 = nn.LSTM(128, 128, bidirectional=True, batch_first=True, num_layers=2)
        #self.L4 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.L5= nn.Linear(256, 90)

    def forward(self, inputs,  seq_lens):
        total_length = inputs.shape[1]
        output = self.L1(inputs)
        output = F.relu(self.L2(output))

        x_packed = pack_padded_sequence(output, seq_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.L3(x_packed)
        #output, _ = self.L4(output)
        output, length = pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = self.L5(output)
        return output


class TransB1(nn.Module):
    def __init__(self):
        super(TransB1, self).__init__()
        """
        TransB1
        输入x1
        输入支撑脚的概率
        """
        self.L1 = nn.Dropout(p=0.2)
        self.L2 = nn.Linear(87, 64)  # relu
        self.L3 = nn.LSTM(64, 64, bidirectional=True, batch_first=True, num_layers=2)
        #self.L4 = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        self.L5= nn.Linear(128, 2)
        self.af = nn.Sigmoid()
    def forward(self, inputs,  seq_lens):
        total_length = inputs.shape[1]
        output = self.L1(inputs)
        output = F.relu(self.L2(output))

        x_packed = pack_padded_sequence(output, seq_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.L3(x_packed)
        #output, _ = self.L4(output)
        output, length = pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = self.af(self.L5(output))
        return output

class TransB2(nn.Module):
    def __init__(self):
        super(TransB2, self).__init__()
        """
        TransB2
        输入x2
        输入根位移
        """
        self.L1 = nn.Dropout(p=0.2)
        self.L2 = nn.Linear(141, 256)  # relu
        self.L3 = nn.LSTM(256, 256, bidirectional=False, batch_first=True, num_layers=2)
        #self.L4 = nn.LSTM(256, 256, bidirectional=False, batch_first=True)
        self.L5= nn.Linear(256, 3)
    def forward(self, inputs,  seq_lens):
        total_length = inputs.shape[1]
        output = self.L1(inputs)
        output = F.relu(self.L2(output))

        x_packed = pack_padded_sequence(output, seq_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.L3(x_packed)
        #output, _ = self.L4(output)
        output, length = pad_packed_sequence(output, batch_first=True, total_length=total_length)

        output = self.L5(output)
        return output