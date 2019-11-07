import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torchvision import transforms
import cv2
import copy
import math
import os
import random
import vlc
playlist = [
    '/Users/arghachakraborty/Desktop/1.mp4', '/Users/arghachakraborty/Desktop/2.mp4','/Users/arghachakraborty/Desktop/3.MP4'
    ]
player = vlc.MediaPlayer(playlist[0])

PATH = '/Users/arghachakraborty/Projects/CV_assignments/Assignments/Assignment_4/trained_smart2.pth'
transform = transforms.Compose([
transforms.ToPILImage(),            
transforms.Scale((50,50)),                   
transforms.ToTensor()                    
 ])

import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc0 = nn.Linear(10 * 24 * 24, 600)
#         self.fc1 = nn.Linear(600, 300)
#         self.fc2 = nn.Linear(300, 20)
#         self.fc3 = nn.Linear(20, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = x.view(-1, 10 * 24 * 24)
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc0 = nn.Linear(6 * 24 * 24, 600)
#         self.fc1 = nn.Linear(600, 300)
#         self.fc2 = nn.Linear(300, 20)
#         self.fc3 = nn.Linear(20, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = x.view(-1, 6 * 24 * 24)
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc0 = nn.Linear(8 * 24 * 24, 800)
#         self.fc1 = nn.Linear(800, 300)
#         self.fc2 = nn.Linear(300, 20)
#         self.fc3 = nn.Linear(20, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = x.view(-1, 8 * 24 * 24)
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1936, 256)
        self.fc2 = nn.Linear(256, 20)
        self.fc3 = nn.Linear(20, 3)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1936)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



#print(c1.

net = Net()
net.load_state_dict(torch.load(PATH))
net.eval()

classes = ['stop','next', 'prev']


cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
 
# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
prev_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/test/prev/'
stop_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/test/stop/'
next_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/test/next/'
prevSwitch = False
stopSwitch = False
nextSwitch = False
play_pause = False
 
def count_files(in_directory):
    joiner= (in_directory + os.path.sep).__add__
    return sum(
        os.path.isfile(filename)
        for filename
        in map(joiner, os.listdir(in_directory))
    )


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
 
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res
 
def clear_all_switch():
    prevSwitch = False
    stopSwitch = False
    nextSwitch = False
 
def capture_frame(frame):
    global count_prv, count_stp, count_nxt, prevSwitch, stopSwitch, nextSwitch
    frame = cv2.resize(frame, (50, 50), interpolation = cv2.INTER_AREA)
    if prevSwitch == True:
        count_prv = count_prv + 1
        cv2.imwrite(prev_1+str(count_prv)+".jpeg", frame)
        print("prev "+ str(count_prv))
    elif stopSwitch == True:
        count_stp = count_stp + 1
        cv2.imwrite(stop_1+str(count_stp)+".jpeg", frame)
        print("stop "+ str(count_stp))
    elif nextSwitch == True:
        count_nxt = count_nxt + 1
        cv2.imwrite(next_1+str(count_nxt)+".jpeg", frame)
        print("next "+ str(count_nxt))

def play():
    if play_pause == True:
        player.play()
    else:
        player.pause()    
 
# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
# cv2.namedWindow('trackbar')
# cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
count_prv = count_files(prev_1)
count_nxt = count_files(next_1)
count_stp = count_files(stop_1)
state = False
infer = [0.0, 0.0, 0.0]
label = 'prev'
epsilon = 0.1
while camera.isOpened():
    ret, frame = camera.read()
    # threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 1)
    cv2.imshow('original', frame)
    old_label = label
    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        frame_ = copy.deepcopy(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        frame_ = frame_[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)
 
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('ori', thresh)

        # thresh = cv2.putText(thresh, classes[index[0]] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                #    1, (0,255,0), 2, cv2.LINE_AA) 
        frame_ = cv2.bitwise_and(frame_,frame_,mask = thresh)
        img_t = transform(frame_)
        batch_t = torch.unsqueeze(img_t, 0)
        net.eval()
        out = net(batch_t)
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # print(classes[index[0]] + ' ' + str(percentage[index[0]].item()))
        # print(percentage)
        label = classes[index[0]]
        if label == old_label : 
            if label == 'prev':
                infer[0] = np.clip(infer[0] + epsilon, 0.0, 1.0)
                infer[1] = np.clip(infer[1] - epsilon, 0.0, 1.0)
                infer[2] = np.clip(infer[2] - epsilon, 0.0, 1.0)
            elif label == 'stop':
                infer[0] = np.clip(infer[0] - epsilon, 0.0, 1.0)
                infer[1] = np.clip(infer[1] + epsilon, 0.0, 1.0)
                infer[2] = np.clip(infer[2] - epsilon, 0.0, 1.0)
            elif label == 'next':
                infer[0] = np.clip(infer[0] - epsilon, 0.0, 1.0)
                infer[1] = np.clip(infer[1] - epsilon, 0.0, 1.0)
                infer[2] = np.clip(infer[2] + epsilon, 0.0, 1.0)
        index_max = np.argmax(infer)
        print(infer)
        # index[0] = index_max
        frame_ = cv2.putText(frame_, classes[index_max] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                1, (0,255,0), 2, cv2.LINE_AA) 
        # frame_ = cv2.putText(frame_, classes[index[0]] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
        #            1, (0,255,0), 2, cv2.LINE_AA) 
        # if classes[index[0]] == 'stop':
        #     play_pause = not play_pause
        #     play()

        
        cv2.imshow('Frame',frame_ )

        if (prevSwitch or stopSwitch or nextSwitch ) and random.randint(1,101) < 40:
            capture_frame(frame_)

 
 
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        clear_all_switch()
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
    elif k == ord('u'):
        clear_all_switch()
        prevSwitch = True
        print ('Anotating prev')
    elif k == ord('i'):
        clear_all_switch()
        stopSwitch = True
        print ('Anotating stop')
    elif k == ord('o'):
        clear_all_switch()
        nextSwitch = True
        print ('Anotating next')
    elif k == ord('p'):
        clear_all_switch()
        print ('clearall')

# print(count_files(stop_1))