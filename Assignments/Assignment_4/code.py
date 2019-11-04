


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torchvision.transforms as transforms
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# # import matplotlib.pyplot as plt
# import time
# import os
# import copy

# # plt.ion() 
# import cv2
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# transform = transforms.Compose([
# transforms.ToPILImage(),            
# transforms.Scale((50,50)),                    
# transforms.ToTensor()                    
#  ])
# classes = ['next', 'prev', 'stop']

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16 * 11 * 11, 200)
#         self.fc2 = nn.Linear(200, 40)
#         self.fc3 = nn.Linear(40, 3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 11 *11)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# # cap = cv2.VideoCapture('/Users/arghachakraborty/Desktop/stop_1.mov')
# # live camera feed
# cap = cv2.VideoCapture('/Users/arghachakraborty/Desktop/stop_1.mov')
# PATH = '/Users/arghachakraborty/Projects/CV_assignments/Assignments/Assignment_4/trained_net_3.pth'
# net = Net()
# net.load_state_dict(torch.load(PATH))
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     img_t = transform(frame)
#     batch_t = torch.unsqueeze(img_t, 0)
#     net.eval()
#     out = net(batch_t)
#     _, index = torch.max(out, 1)
#     percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     # print(classes[index[0]], percentage[index[0]].item())
#     frame = cv2.putText(frame, classes[index[0]] + ' ' + str(percentage[index[0]].item()), (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
#                    1, (0,255,0), 2, cv2.LINE_AA) 
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("test")
# *((*((*(*(*(*(*(*((*(*(*(*))))))))))))))
# Required moduls
import cv2
import numpy
# from pykeyboard import PyKeyboard

# Constants for finding range of skin color in YCrCb
# min_YCrCb = numpy.array([0,96,107],numpy.uint8)
# max_YCrCb = numpy.array([134,255,123],numpy.uint8)
min_YCrCb = numpy.array([0,96,103],numpy.uint8)
max_YCrCb = numpy.array([134,255,123],numpy.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
gameflag = False
# keyboard = PyKeyboard()

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)
# videoFrame = cv2.VideoCapture('/Users/arghachakraborty/Downloads/sukhraj_train_vids/stop.mp4')

while True: # any key pressed has a value >= 0

    # Grab video frame, decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()
    sourceImage = cv2.flip(sourceImage, 1)

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    
    skin = cv2.bitwise_and(sourceImage, sourceImage, mask = skinRegion)

    # Do contour detection on skin region
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.rectangle(skin,(0,300),(640,200),(0,0,255),3)
    # cv2.drawContours(skin, contours, -1, (0,255,0), 3)
    # Display the source image
    final = cv2.medianBlur(skin, 7)
    cv2.imshow('Camera Output',final)
    # cv2.imwrite('test.jpeg', skin)

    # Check for user input to close program
    if cv2.waitKey(5) == 27: # wait 5 millisecond in each iteration of while loop
        break 

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# def nothing(x):
#   pass
# cv2.namedWindow('Colorbars')
# hh='Max'
# hl='Min'
# wnd = 'Colorbars'
# cv2.createTrackbar("Min_y", "Colorbars",0,255,nothing)
# cv2.createTrackbar("Min_cb", "Colorbars",0,255,nothing)
# cv2.createTrackbar("Min_cr", "Colorbars",0,255,nothing)
# cv2.createTrackbar("Max_y", "Colorbars",0,255,nothing)
# cv2.createTrackbar("Max_cb", "Colorbars",0,255,nothing)
# cv2.createTrackbar("Max_cr", "Colorbars",0,255,nothing)
# img = cv2.imread('test.jpeg',1)
# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
# cv2.imshow(' Output',img)

# while(1):
#    miny=cv2.getTrackbarPos("Min_y", "Colorbars")
#    mincb=cv2.getTrackbarPos("Min_cb", "Colorbars")
#    mincr=cv2.getTrackbarPos("Min_cr", "Colorbars")   
#    maxy=cv2.getTrackbarPos("Max_y", "Colorbars")
#    maxcb=cv2.getTrackbarPos("Max_cb", "Colorbars")
#    maxcr=cv2.getTrackbarPos("Max_cr", "Colorbars")
#    print("miny" + str(miny))
#    print("mincb" + str(mincb))
#    print("mincr" + str(mincr))
#    print("maxy" + str(maxy))
#    print("maxcb" + str(maxcb))
#    print("maxcr" + str(maxcr))
#    imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
#    min_YCrCb = np.array([miny,mincb,mincr],np.uint8)
#    max_YCrCb = np.array([maxy,maxcb,maxcr],np.uint8)
#     # Find region with skin tone in YCrCb image
#    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
#    skin = cv2.bitwise_and(img, img, mask = skinRegion)
#    cv2.imshow('Camera Output',skin)
#    k = cv2.waitKey(1) & 0xFF
#    if k == ord('m'):
#      mode = not mode
#    elif k == 27:
#      break
# cv2.destroyAllWindows()