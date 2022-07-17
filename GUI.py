import cv2
import torch, torchvision
import numpy as np
import pandas as pd
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import PIL.Image as Image
from PIL import Image, ImageTk
import threading
import tkinter as tk
import shutil
import time
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models



videoloop_stop = [False]
global class_idx
class_idx = [43]
global model
global show
import os
show = [False]
'''
class_names = [
    'Limite velocità 20km/h', 'Limite velocità 30km/h', 'Limite velocità 50km/h', 
    'Limite velocità 60km/h', 'Limite velocità 70km/h', 'Limite velocità 80km/h', 
    'Fine limite velocità 80km/h', 'Limite velocità 100km/h', 
    'Limite velocità 120km/h', 'Divieto di sorpasso', 
    'Divieto sorpasso per veicoli oltre 3.5 tonn', 
    "Diritto precedenza all' incrocio", 'Strada prioritaria', 'Dare precedenza', 
    'Stop', 'Divieto di transito', 'Divieto di transito Veicoli oltre 3.5 tonn', 
    'Senso vietato', 'Pericolo generico', 'Curva pericolosa a sinistra', 
    'Curva pericolosa a destra', 'Doppia curva', 'Dossi', 
    'Strada sdrucciolevole', 'Restringimento carreggiata destra', 'Cantieri stradali', 
    'Semaforo', 'Attraversamento pedonale', 'Attraversamento bambini', 
    'Attraversamento ciclabile', 'Pericolo ghiaccio/neve', 'Attraversamento animali selvatici', 
    'Fine di tutti i limiti di velocità e sorpasso', 'Obbligo svolta a destra avanti', 
    'Svolta a sinistra avanti', 'Obbligo diritto', 'Obbligo diritto o destra', 
    'Obbligo diritto o sinistra', 'Mantieni la destra', 'Mantieni la sinistra', 'Rotatoria', 
    'Fine del divieto di sorpasso', 'Fine divieto di sorpasso Veicoli oltre 3.5 tonn', ''
]'''

class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)', 
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 
    'No passing for vehicles over 3.5 metric tons', 
    'Right-of-way at next intersection', 'Priority road', 'Yield', 
    'Stop', 'No vehicles', 'Vehicles over 3.5 tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve to the left', 
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 
    'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children crossing', 
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 
    'End of all speed and passing limits', 'Turn right ahead', 
    'Turn left ahead', 'Ahead only', 'Go straight or right', 
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory', 
    'End of no passing', 'End of no passing vehicles over 3.5 tons', ''
]
n_classes = len(class_names) - 1

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

class_idx = 43


def startButton_clicked(videoloop_stop):
    startButton.configure(bg="green", fg="white")
    stopButton.configure(bg="gray", fg="black")
    videoloop_stop[1]=='nessuna'
    threa = threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()


def stopButton_clicked(videoloop_stop):
    stopButton.configure(bg="red", fg="white")
    startButton.configure(bg="gray", fg="black")
    videoloop_stop[0] = True
    videoloop_stop[1]=='nessuna'


def resnet18_no_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="green", fg="white")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")

  
    videoloop_stop[1] = 'resnet18_no'
    startButton_clicked(videoloop_stop)

def resnet18_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="green", fg="white")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")
  

    videoloop_stop[1] = 'resnet18'
    startButton_clicked(videoloop_stop)

def alexnet_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="green", fg="white")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")
  

    videoloop_stop[1] = 'alexnet'
    startButton_clicked(videoloop_stop)

def googleLeNet_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="green", fg="white")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")


    videoloop_stop[1] = 'googleLeNet'
    startButton_clicked(videoloop_stop)

def shufflenet_v2_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="green", fg="white")
  

    videoloop_stop[1] = 'shufflenet_v2'
    startButton_clicked(videoloop_stop)

def mobilenet_v2_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="green", fg="white")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")
  

    videoloop_stop[1] = 'mobilenet_v2'
    startButton_clicked(videoloop_stop)

def mobilenet_v3_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="green", fg="white")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")
  

    videoloop_stop[1] = 'mobilenet_v3'
    startButton_clicked(videoloop_stop)

def efficientnet_b0_clicked(videoloop_stop):
    stopButton_clicked(videoloop_stop)
    
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="green", fg="white")
    shufflenet_v2_Button.configure(bg="white", fg="black")

    
    videoloop_stop[1] = 'efficientnet_b0'
    startButton_clicked(videoloop_stop)

def info_clicked(videoloop_stop):
    videoloop_stop[1] = 'info'
    resnet18_no_Button.configure(bg="white", fg="black")
    resnet18_Button.configure(bg="white", fg="black")
    alexnet_Button.configure(bg="white", fg="black")
    googleLeNet_Button.configure(bg="white", fg="black")
    mobilenet_v2_Button.configure(bg="white", fg="black")
    mobilenet_v3_Button.configure(bg="white", fg="black")
    efficientnet_b0_Button.configure(bg="white", fg="black")
    shufflenet_v2_Button.configure(bg="white", fg="black")

    image = ImageTk.PhotoImage(filename='architectures-plot.png')
    panel = tk.Label(root, image=image, justify=tk.LEFT, padx = 20, text='Info', compound='bottom')
    panel.config(bg="yellow")
    panel.config(font=("Courier", 24))
    panel.config(fg="#FFFFFF")
    panel.image = image
    panel.place(x=50, y=50)
'''
    from tkinter import *
root=Tk()
img=PhotoImage(file='sunshine.jpg')
Label(root,image=img).pack()
root.mainloop()

'''
def openInfoWindow():
    '''
    filename = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select file", filetypes=(("png images", ".png"), ("all files", "*.*")))
    if not filename:
        return'''
    # setup new window
    new_window = Toplevel(root)
    # get image
    image = ImageTk.PhotoImage(Image.open('C:/Users/admin/Desktop/GUI_tesi/architectures-plot.png'))
    # load image
    panel = Label(new_window, image=image)
    panel.image = image
    panel.pack()


def videoLoop(mirror=True):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    if(videoloop_stop[1] == 'resnet18_no' or videoloop_stop[1] == 'nessuna'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_backup/resnet18_vecchio_model.pt', map_location=device)
      modello.eval()
    elif(videoloop_stop[1] == 'resnet18'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_backup/resnet18_vecchio_model.pt', map_location=device)
      modello.eval()
    elif(videoloop_stop[1] == 'alexnet'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/alexnet/alexnet_rumore_model.pt', map_location=device)
      modello.eval()
    
    elif(videoloop_stop[1] == 'googleLeNet'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/googleLeNet/googleLeNet_model.pt', map_location=device)
      modello.eval()
      
    elif(videoloop_stop[1] == 'shufflenet_v2'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/shufflenet_v2/shufflenet_v2_model.pt', map_location=device)
      modello.eval()
      '''
    elif(videoloop_stop[1] == 'mobilenet_v2'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/mobilenet_v2/mobilenet_v2_model.pt', map_location=device)
      modello.eval()
    '''
    elif(videoloop_stop[1] == 'mobilenet_v3'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/mobilenet_v3/mobilenet_v3_model.pt', map_location=device)
      modello.eval()
      
    elif(videoloop_stop[1] == 'efficientnet_b0'):
      modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/efficientnet_b0/efficientnet_b0_model.pt', map_location=device)
      modello.eval()
    
    while True:
        ret, to_draw = cap.read()
        if mirror is True:
            to_draw = to_draw[:, ::-1]

        image = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        correct_count = 0
        frame_count = 0 # To count total frames.
        total_fps = 0 # To get the final frames per second. 
        test_images = 0
        #if pred[0] == True:
        if True:
            #image2 = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
            #image2 = Image.fromarray(image2)

            #Predizione modello
            image2 = transforms['test'](image).unsqueeze(0) 

            
            #modello.eval()
            start_time = time.time()
            pred = modello(image2.to(device))
            end_time = time.time()
            pred = F.softmax(pred, dim=1)
            _, class_idx[0] = torch.max(pred,1)

            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            '''
            #Scrittura su frame mostrato
            label_font = cv2.FONT_HERSHEY_DUPLEX
            class_idx = convert(class_idx)
            print(class_idx)
            cv2.putText(img=frame,
                    #text=class_names[class_idx]+" "+f"{fps:.1f} FPS",
                    text=class_names[class_idx], org=(30,450),
                    fontFace=label_font, #fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1.5, color=(0, 255, 0),thickness=1)
            '''

        
        #label = tk.Label(root, text='Funziona', image=image, compound='center')
        #label.pack()
        #tk.Label(root, image=image, text="Update User",
                 #compound=tk.CENTER).pack() # Put it in the display window
        if(videoloop_stop[1]=='nessuna'):
          image = ImageTk.PhotoImage(image)
          panel = tk.Label(root, image=image, justify=tk.LEFT, padx = 20, text='Scegliere una rete', compound='bottom')
          panel.config(bg="red")
        elif(videoloop_stop[1]=='info'):
          image = ImageTk.PhotoImage(Image.open('architectures-plot.png'))
          panel = tk.Label(root, image=image, justify=tk.LEFT, padx = 10, text='Info', compound='bottom')
          panel.config(bg="yellow")
        else:
          image = ImageTk.PhotoImage(image)
          testo = class_names[class_idx[0]]+'\nFPS: '+str(round(total_fps))
          panel = tk.Label(root, image=image, justify=tk.LEFT, padx = 30, text=testo, compound='bottom')
          panel.config(bg="green")
        panel.config(font=("Courier", 24))
        panel.config(fg="#FFFFFF")
        panel.image = image
        panel.place(x=50, y=50)

        # check switcher value
        if videoloop_stop[0]:
            # if switcher tells to stop then we switch it again and stop videoloop
            videoloop_stop[0] = False
            panel.destroy()
            break


# videoloop_stop is a simple switcher between ON and OFF modes
videoloop_stop = [False, 'nessuna']

class_idx = [43]
model = []
root = tk.Tk()
root.title('Traffic Sign Recognition inference')
root.geometry("1920x1080+0+0")
#root.geometry("3840x2169+0+0")
'''
canvas = Canvas(
    root,
    height=1080,
    width=1920,
    bg=None,
    )

canvas.pack()
'''

#START
startButton = tk.Button(
    root, text="Start", bg="#fff", font=("", 20),
    command=lambda: startButton_clicked(videoloop_stop))
startButton.place(x=976, y=50, width=140, height=90)

#STOP
stopButton = tk.Button(
    root, text="Stop", bg="#fff", font=("", 20),
    command=lambda: stopButton_clicked(videoloop_stop))
stopButton.place(x=1124, y=50, width=140, height=90)

#RESNET18_NO
resnet18_no_Button = tk.Button(
    root, text="resnet18_no", bg="#fff", font=("", 15),
    command=lambda: resnet18_no_clicked(videoloop_stop))
resnet18_no_Button.place(x=980, y=250, width=140, height=80)

#RESNET18
resnet18_Button = tk.Button(
    root, text="resnet18", bg="#fff", font=("", 15),
    command=lambda: resnet18_clicked(videoloop_stop))
resnet18_Button.place(x=1120, y=250, width=140, height=80)

#ALEXNET
alexnet_Button = tk.Button(
    root, text="alexnet", bg="#fff", font=("", 15),
    command=lambda: alexnet_clicked(videoloop_stop))
alexnet_Button.place(x=980, y=330, width=140, height=80)

#GOOGLELENET
googleLeNet_Button = tk.Button(
    root, text="googleLeNet", bg="#fff", font=("", 15),
    command=lambda: googleLeNet_clicked(videoloop_stop))
googleLeNet_Button.place(x=1120, y=330, width=140, height=80)

#MOBILENET_v2
mobilenet_v2_Button = tk.Button(
    root, text="mobilenet_v2", bg="#fff", font=("", 15),
    command=lambda: mobilenet_v2_clicked(videoloop_stop))
mobilenet_v2_Button.place(x=980, y=410, width=140, height=80)

#MOBILENET_v3
mobilenet_v3_Button = tk.Button(
    root, text="mobilenet_v3", bg="#fff", font=("", 15),
    command=lambda: mobilenet_v3_clicked(videoloop_stop))
mobilenet_v3_Button.place(x=1120, y=410, width=140, height=80)

#efficientnet_b0
efficientnet_b0_Button = tk.Button(
    root, text="efficientnet_b0", bg="#fff", font=("", 15),
    command=lambda: efficientnet_b0_clicked(videoloop_stop))
efficientnet_b0_Button.place(x=980, y=490, width=140, height=80)

#sufflenet_v2
shufflenet_v2_Button = tk.Button(
    root, text="shufflenet_v2", bg="#fff", font=("", 15),
    command=lambda: shufflenet_v2_clicked(videoloop_stop))
shufflenet_v2_Button.place(x=1120, y=490, width=140, height=80)

#info
info_Button = tk.Button(
    root, text="info", bg="#fff", font=("",15),
    command=lambda: openInfoWindow())
info_Button.place(x=1160, y=580, width=100, height=50)

root.mainloop()
