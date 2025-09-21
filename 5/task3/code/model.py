import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__ (self, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3) # Conv 1 layer
        self.bn1 = nn.BatchNorm2d(16) # BatchNorm layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)# Conv 2 layer
        self.bn2 = nn.BatchNorm2d(32) # BatchNorm layer
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)# Conv 3 layer
        self.bn3 = nn.BatchNorm2d(48) # BatchNorm layer
        self.conv4 =nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)# Conv 4 layer
        self.bn4 = nn.BatchNorm2d(64) # BatchNorm layer
        self.conv5 = nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=1) # Conv 5 layer
        self.relu = nn.ReLU() # ReLU layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # Avgpool layer
        self.fc = nn.Linear(80, num_classes) # Linear layer
    
        
    def forward(self, x, intermediate_outputs=False):
        # TODO: 按照题目描述中的示意图计算前向传播的输出，如果 intermediate_outputs 为 True，也返回各个卷积层的输出。
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = None, None, None, None, None
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if intermediate_outputs:
            conv1_out = x.clone()
            
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if intermediate_outputs:
            conv2_out = x.clone()
            
        x = self.conv3(x)
        x = self.bn3(x) 
        x = self.relu(x)
        x = self.maxpool(x)
        if intermediate_outputs:
            conv3_out = x.clone()
            
        x = self.conv4(x)
        x = self.bn4(x)  
        x = self.relu(x)
        x = self.maxpool(x)
        if intermediate_outputs:
            conv4_out = x.clone()
            
        x = self.conv5(x)
        x = self.avgpool(x)
        if intermediate_outputs:
            conv5_out = x.clone()
            
        x = x.view(x.size(0), -1)
        final_out = self.fc(x)
            
        if intermediate_outputs:
            return final_out, conv1_out, conv2_out, conv3_out, conv4_out, conv5_out
        else:
            return final_out
