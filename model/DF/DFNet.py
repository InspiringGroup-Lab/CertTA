import torch
import torch.nn as nn


class DFNet(nn.Module):
    def __init__(self, input_shape, classes) -> None:
        super(DFNet, self).__init__()
        # input_shape = [2, LENGTH]

        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        
        if input_shape[1] == 5000:
            pool_stride_size = ['None',4,4,4,4]
            pool_size = ['None',8,8,8,8]
            fc1_linear_in_dim = 4608
        elif input_shape[1] <= 100:
            pool_stride_size = ['None',2,2,2,2]
            pool_size = ['None',4,4,4,4]
            fc1_linear_in_dim = 1024
        else:
            raise Exception('Wrong LENGTH!')

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding='same'),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding='same'),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1]),
            nn.Dropout(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding='same'),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding='same'),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2]),
            nn.Dropout(0.1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding='same'),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding='same'),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3]),
            nn.Dropout(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding='same'),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding='same'),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4]),
            nn.Dropout(0.1)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=fc1_linear_in_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=512, out_features=classes)
        )
    
    def forward(self, x):
        # x: [batch_size, 2, LENGTH]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.out(x)
        return logits
