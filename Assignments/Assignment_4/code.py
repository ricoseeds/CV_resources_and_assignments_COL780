


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy

# plt.ion() 
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
transform = transforms.Compose([
transforms.ToPILImage(),            
transforms.Scale((50,50)),                    
transforms.ToTensor()                    
 ])
classes = ['next', 'prev', 'stop']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 11 * 11, 200)
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 11 *11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cap = cv2.VideoCapture('/Users/arghachakraborty/Desktop/stop_1.mov')
PATH = '/Users/arghachakraborty/Projects/CV_assignments/Assignments/Assignment_4/trained_net_3.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
while(cap.isOpened()):
    ret, frame = cap.read()
    img_t = transform(frame)
    batch_t = torch.unsqueeze(img_t, 0)
    net.eval()
    out = net(batch_t)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # print(classes[index[0]], percentage[index[0]].item())
    frame = cv2.putText(frame, classes[index[0]] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,255,0), 2, cv2.LINE_AA) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("test")