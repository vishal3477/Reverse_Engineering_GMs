##########################
### MODEL
##########################
import torch
import torch.nn as nn
class encoder(torch.nn.Module):

    def __init__(self,num_hidden):
        super(encoder, self).__init__()
        
        ### ENCODER
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(65536, num_hidden)
        self.fc2=nn.Linear(num_hidden, 5)
        self.relu=nn.ReLU()
        

    def forward(self, x):
        
        ### ENCODER
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out=self.relu(out)
        out1=self.fc2(out)
        out1 = torch.sigmoid(out1)
        return out1, out
