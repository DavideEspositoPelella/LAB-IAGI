
#import face_recognition
import torch
import time
import argparse
import pathlib
import PIL
from PIL import Image
import torch, torchvision
from pathlib import Path
import numpy as np
import pandas as pd
import PIL.Image as Image
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from torch.optim import lr_scheduler
from glob import glob
import shutil
from collections import defaultdict
import os
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import cv2 # pip install opencv-python
from imutils.video import VideoStream
from imutils.video import FPS

class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 
    'No passing for vehicles over 3.5 metric tons', 
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 
    'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve to the left', 
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 
    'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 
    'Turn left ahead', 'Ahead only', 'Go straight or right', 
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]
n_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
transforms = {'train': T.Compose([
  T.RandomResizedCrop(size=256),
  T.RandomAffine(degrees=(-20, 20), translate=(0.1,0.3), scale=(0.7,1), shear=(-20, 20)),
  #T.RandomErasing(), #cancella autonomamente parti dell'immagine
  #AddGaussianNoise(0.1, 0.08),  #rumore gaussiano (vero e proprio "rumore del sensore", disturbo)
  T.ColorJitter(brightness=0.7, contrast=0.3, saturation=0.4, hue=0), #modifica condizioni foto
  T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #blur
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)

]), 'val': T.Compose([
  T.Resize(size=256),
  T.CenterCrop(size=224),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]), 'test': T.Compose([
  T.Resize(size=256),
  T.CenterCrop(size=224),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]),
}

"""##Funzione Conversione"""

def convert(pred):
  
  if pred == 0: #ahead
    return 35  
  elif pred == 1:  #beware of ice
    return 30
  elif pred == 2: #bicycles crossing
    return 29
  elif pred == 3:  #bumpy road
    return 22
  elif pred == 4:  #children crossing
    return 28
  elif pred == 5:  #dangerous curve to the left
    return 19
  elif pred == 6:  #dangerous curve to the right
    return 20
  elif pred == 7:  #Double curve
    return 21
  elif pred == 8:  #End of all speed and passing limits
    return 32
  elif pred == 9:  #End of no passing
    return 41
  elif pred == 10: #End of no passing by vehicles over 3.5 metric tons
    return 42
  elif pred == 11: #End of speed limit (80km/h)
    return 6
  elif pred == 12: #General caution
    return 18
  elif pred == 13: #Go straight or left
    return 37
  elif pred == 14: #Go straight or right
    return 36
  elif pred == 15: #Keep left
    return 39
  elif pred == 16: #Keep right
    return 38
  elif pred == 17: #No entry
    return 17
  elif pred == 18: #No passing
    return 9
  elif pred == 19: #No passing for vehicles over 3.5 metric tons
    return 10
  elif pred == 20: #No vehicles
    return 15
  elif pred == 21: #Pedestrians
    return 27
  elif pred == 22: #Priority road
    return 12
  elif pred == 23: #Right-of-way at the next intersection
    return 11
  elif pred == 24: #Road narrows on the right
    return 24
  elif pred == 25: #Road work
    return 25
  elif pred == 26: #Roundabout mandatory
    return 40
  elif pred == 27: #Slippery road
    return 23
  elif pred == 28: #Speed limit (100km/h)
    return 7
  elif pred == 29: #Speed limit (120km/h)
    return 8
  elif pred == 30: #Speed limit (20km/h)
    return 0
  elif pred == 31: #Speed limit (30km/h)
    return 1
  elif pred == 32: #Speed limit (50km/h)
    return 2
  elif pred == 33: #Speed limit (60km/h)
    return 3
  elif pred == 34: #Speed limit (70km/h)
    return 4
  elif pred == 35: #Speed limit (80km/h)
    return 5
  elif pred == 36: #Stop
    return 14
  elif pred == 37: #Traffic Signals
    return 26
  elif pred == 38: #Turn left ahead
    return 34
  elif pred == 39: #Turn right ahead
    return 33
  elif pred == 40: #Vehicles over 3.5 metric tons prohibited
    return 16
  elif pred == 41: #Wild animals crossing
    return 31
  elif pred == 42: #Yield(give_way)
    return 13

def create_model(net_model, n_classes):

  if(net_model=='resnet18_no'):
    model= models.resnet18(pretrained = False, progress = True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

  if(net_model=='resnet18'):
    model = models.resnet18(pretrained = True, progress = True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)  

  if (net_model == 'alexnet'):
    model = models.alexnet(pretrained = True, progress = True)
    n_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_features,n_classes)

  if(net_model == 'googleLeNet'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

  if(net_model=='mobilenet_v2'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)

  if(net_model=='mobilenet_v3'):
    model = models.mobilenet_v3_small(pretrained=True, progress=True)
    model.classifier[-1] = nn.Linear(1024, n_classes)

  if(net_model == 'shufflenet_v2'):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
    model.fc = nn.Linear(1024, n_classes)

  if(net_model == 'efficientnet_b0'):
    model = models.efficientnet_b0(pretrained=True, progress=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes)
  return model.to(device)


'''  
resnet18_no = create_model("resnet18_no",len(class_names))
#resnet18_no.load_state_dict(torch.load('/Modelli/resnet18_no/resnet18_no_pretrain_no_augmentation.pt', map_location = device))     
resnet18_no.load_state_dict(torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_no/resnet18_no.pt' , map_location=device))
resnet18_no.eval()
'''
#resnet18 = create_model("resnet18",len(class_names))
#resnet18.load_state_dict(torch.load('/Modelli/resnet18/resnet18.pt', map_location = device)) 
#resnet18.eval()

resnet18 = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18/resnet18_model.pt', map_location=device)
resnet18.eval()

'''
alexnet = create_model("alexnet",len(class_names))
alexnet.load_state_dict(torch.load('/Modelli/alexnet/alexnet.pt', map_location = device)) 
alexnet.eval()

googleLeNet = create_model("googleLeNet",len(class_names))
googleLeNet.load_state_dict(torch.load('/Modelli/googleLeNet/googleLeNet.pt', map_location = device)) 
googleLeNet.eval()

 
mobilenet_v2 = create_model("mobilenet_v2",len(class_names))
mobilenet_v2.load_state_dict(torch.load('/Modelli/mobilenet_v2/mobilenet_v2.pt', map_location = device)) 
mobilenet_v2.eval()
  
mobilenet_v3 = create_model("mobilenet_v3",len(class_names))
mobilenet_v3.load_state_dict(torch.load('/Modelli/mobilenet_v3/mobilenet_v3.pt', map_location = device)) 
mobilenet_v3.eval()
  
shufflenet_v2 = create_model("shufflenet_v2",len(class_names))
shufflenet_v2.load_state_dict(torch.load('/Modelli/shufflenet_v2/shufflenet_v2.pt', map_location = device)) 
shufflenet_v2.eval()
  
efficientnet_b0 = create_model("efficientnet_b0",len(class_names))
efficientnet_b0.load_state_dict(torch.load('/Modelli/efficientnet_b0/efficientnet_b0.pt', map_location = device)) 
efficientnet_b0.eval()
'''

webcam = cv2.VideoCapture(0)

#image_file = input("Target Image File > ")
#target_image = face_recognition.load_image_file(image_file)
#target_encoding = face_recognition.face_encodings(target_image)[0]

print("Webcam Inizializzata \n")

#target_name = input("Target Name > ")  #input atteso prima di proseguire

process_this_frame = True  #processa un frame ogni 2
#Modello
model = resnet18
#model.eval()
frame_count = 0 
total_fps = 0
i = 0
while webcam.isOpened():
    i=i+1
    ret, frame = webcam.read()

    if not ret:
      break
    
    if process_this_frame:
        image = frame.copy()
        #image = cv2.resize(image, None, fx=0.20, fy=0.20)

        image2 =Image.fromarray(image)
        #image /= 255.0
        image2 = image2.convert('RGB')
        
        image2 = transforms['test'](image2).unsqueeze(0)

        #start_time = time.time()
        #with torch.no_grad():
        pred = model(image2.to(device))
        #end_time = time.time()        
        pred = F.softmax(pred, dim=1)
        _, class_idx = torch.max(pred,1)


        #fps = 1 / (end_time - start_time)
        #total_fps += fps
        #frame_count += 1
    #if(i%3==0):
      #process_this_frame = not process_this_frame

    label_font = cv2.FONT_HERSHEY_DUPLEX
    #cv2.putText(frame, 'finito', (6,- 6), label_font, 0.8, (255, 255, 255), 1)
    
    #class_idx = convert(class_idx)
    print(class_idx)
    cv2.putText(img=frame,
                #text=class_names[class_idx]+" "+f"{fps:.1f} FPS",
                text=class_names[class_idx],
                org=(30, 450), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1.5, color=(0, 255, 0),thickness=1)

    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
