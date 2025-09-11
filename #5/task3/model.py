import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = None # Conv 1 layer
        self.bn1 = None # BatchNorm layer
        self.conv2 = None # Conv 2 layer
        self.bn2 = None # BatchNorm layer
        self.conv3 = None # Conv 3 layer
        self.bn3 = None # BatchNorm layer
        self.conv4 = None # Conv 4 layer
        self.bn4 = None # BatchNorm layer
        self.conv5 = None # Conv 5 layer
        self.relu = None # ReLU layer
        self.maxpool = None # MaxPool layer
        self.avgpool = None # Avgpool layer
        self.fc = None # Linear layer
        self.relu = nn.ReLU()
        self.linear = nn.Linear()
    def forward(self, x, intermediate_outputs=False):
        # TODO: 按照题目描述中的示意图计算前向传播的输出，如果 intermediate_outputs 为 True，也返回各个卷积层的输出。
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.maxpool(x))
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(self.maxpool(x))
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(self.maxpool(x))
        x = self.conv5
        x = self.avgpool(x)
        x = self.linear(x)
        

        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
