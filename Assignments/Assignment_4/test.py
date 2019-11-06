
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

PATH = '/Users/arghachakraborty/Projects/CV_assignments/Assignments/Assignment_4/trained_net_7.pth'
transform = transforms.Compose([
transforms.ToPILImage(),            
transforms.Scale((50,50)),                    
transforms.ToTensor()                    
 ])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 3)
        self.fc1 = nn.Linear(8 * 22 * 22, 100)
        self.fc2 = nn.Linear(100, 30)
        self.fc3 = nn.Linear(30, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * 22 *22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#print(c1.shape())


net = Net()
net
classes = ['next', 'prev', 'stop']


cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
 
# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works
prev_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/train/prev_1/'
stop_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/train/stop_1/'
next_1 = '/Users/arghachakraborty/Projects/CV_assignments/data/train/next_1/'
prevSwitch = False
stopSwitch = False
nextSwitch = False
 
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

 
# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
# cv2.namedWindow('trackbar')
# cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
count_prv = count_files(prev_1)
count_nxt = count_files(next_1)
count_stp = count_files(stop_1)
while camera.isOpened():
    ret, frame = camera.read()
    # threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
 
    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)
 
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('ori', thresh)
        img_t = transform(thresh)
        batch_t = torch.unsqueeze(img_t, 0)
        net.eval()
        out = net(batch_t)
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        thresh = cv2.putText(thresh, classes[index[0]] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,255,0), 2, cv2.LINE_AA) 
        cv2.imshow('Results',thresh)
        print(classes[index[0]] + ' ' + str(percentage[index[0]].item()))
 
        
        # thresh1 = copy.deepcopy(thresh)
        # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # length = len(contours)
        # maxArea = -1
        # if length > 0:
        #     for i in range(length):  # find the biggest contour (according to area)
        #         temp = contours[i]
        #         area = cv2.contourArea(temp)
        #         if area > maxArea:
        #             maxArea = area
        #             ci = i
 
        #     res = contours[ci]
        #     hull = cv2.convexHull(res)
        #     drawing = np.zeros(img.shape, np.uint8)
        #     cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            # cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
 
            # isFinishCal,cnt = calculateFingers(res,drawing)
            # if triggerSwitch is True:
            #     if isFinishCal is True and cnt <= 2:
            #         print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
                   
 
        # cv2.imshow('output', drawing)
 
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