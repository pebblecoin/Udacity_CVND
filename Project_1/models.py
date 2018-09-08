## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Here we want to keep the input and output features the same
        # The formula we can use is
        # Image size is 224*224
        # The shape calculator is 
        # H_out = floor((H_in + 2* padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
        # W_out = floor((W_in + 2* padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 6, padding = 0) # valid 224-6-1 +1 = 218
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5,stride = 1, padding = 0 ) # valid 
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 0) # valid
            
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # output shape = floor((H_in + 2* padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2,stride =2, padding = 0)
            
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features = 73728, out_features = 500)
        self.fc2 = nn.Linear(in_features = 500 , out_features = 136)
        
        ## Double Check the weights initialization
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # Here we need to calculate the shape of all the layers

        # Conv layers
        x = self.maxpool(F.relu(self.conv1(x))) # (218-1-1)/2+1 = 104
        x = self.dropout1(x)

        x = self.maxpool(F.relu(self.conv2(x))) #  conv2 shape = 104 - (4)-1 +1 = 100 maxpool = floor((100-1-1)/2+1) = 50
        x = self.dropout1(x)

        x = self.maxpool(F.relu(self.conv3(x))) # conv2 shape = 50 - 4-1+1 = 46 maxpool = 44/2+1 = 23
        x = self.dropout1(x)

        # Flatten

        x = x.view(x.size(0),-1) #128*23  

        # Fully connected layers

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
