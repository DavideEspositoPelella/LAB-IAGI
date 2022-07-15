import cv2
import torch, torchvision
import numpy as np
import pandas as pd
import PIL.Image as Image
from PIL import ImageTk
import threading
import tkinter as tk
import shutil
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

videoloop_stop = [False]
global pred 
pred = [False]
global class_idx
class_idx = [43]
global model

class_names = [
    'Limite velocità 20km/h', 'Limite velocità 30km/h', 'Limite velocità 50km/h', 
    'Limite velocità 60km/h', 'Limite velocità 70km/h', 'Limite velocità 80km/h', 
    'Fine limite velocità 80km/h', 'Limite velocità 100km/h', 
    'Limite velocità 120km/h', 'Divieto di sorpasso', 
    'Divieto di sorpasso per veicoli oltre 3.5 tonnellate', 
    'Diritto precedenza al prossimo incrocio', 'Strada prioritaria', 'Dare precedenza', 
    'Stop', 'Divieto di transito', 'Divieto di transito per Veicoli oltre 3.5 tonnellate', 
    'Senso vietato', 'Pericolo generico', 'Curva pericolosa a sinistra', 
    'Curva pericolosa a destra', 'Doppia curva', 'Dossi', 
    'Strada sdrucciolevole', 'Restringimento carreggiata destra', 'Cantieri stradali', 
    'Semaforo', 'Attraversamento pedonale', 'Attraversamento bambini', 
    'Attraversamento ciclabile', 'Pericolo ghiaccio/neve', 'Attraversamento animali selvatici', 
    'Fine di tutti i limiti di velocità e sorpasso', 'Obbligo svolta a destra avanti', 
    'Svolta a sinistra avanti', 'Obbligo diritto', 'Obbligo diritto o destra', 
    'Obbligo diritto o sinistra', 'Mantieni la destra', 'Mantieni la sinistra', 'Rotatoria', 
    'Fine del divieto di sorpasso', 'Fine del divieto di sorpasso Veicoli oltre 3.5 tonnellate', ''
]
'''
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
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons', ''
]'''
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

resnet18_no = create_model("resnet18_no",n_classes)     
resnet18_no.load_state_dict(torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_no/resnet18_no.pt' , map_location=device))
resnet18_no.eval()
    
videoloop_stop[1] = True
resnet18 = create_model("resnet18", n_classes)
resnet18.load_state_dict(torch.load('/Modelli/resnet18/resnet18.pt', map_location = device)) 
resnet18.eval()
 
alexnet = create_model("alexnet",n_classes)
alexnet.load_state_dict(torch.load('/Modelli/alexnet/alexnet.pt', map_location = device)) 
alexnet.eval()

googleLeNet = create_model("googleLeNet",n_classes)
googleLeNet.load_state_dict(torch.load('/Modelli/googleLeNet/googleLeNet.pt', map_location = device)) 
googleLeNet.eval()

 
mobilenet_v2 = create_model("mobilenet_v2",n_classes)
mobilenet_v2.load_state_dict(torch.load('/Modelli/mobilenet_v2/mobilenet_v2.pt', map_location = device)) 
mobilenet_v2.eval()
  
mobilenet_v3 = create_model("mobilenet_v3",n_classes)
mobilenet_v3.load_state_dict(torch.load('/Modelli/mobilenet_v3/mobilenet_v3.pt', map_location = device)) 
mobilenet_v3.eval()
  
shufflenet_v2 = create_model("shufflenet_v2",n_classes)
shufflenet_v2.load_state_dict(torch.load('/Modelli/shufflenet_v2/shufflenet_v2.pt', map_location = device)) 
shufflenet_v2.eval()
  
efficientnet_b0 = create_model("efficientnet_b0",n_classes)
efficientnet_b0.load_state_dict(torch.load('/Modelli/efficientnet_b0/efficientnet_b0.pt', map_location = device)) 
efficientnet_b0.eval()
'''



def startButton_clicked(videoloop_stop, pred):
    threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()


def stopButton_clicked(videoloop_stop, pred):
    videoloop_stop[0] = True
    pred[0] = False

def resnet18_no_clicked(videoloop_stop, pred):
    pred[0] = True
    model[0] = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_no/resnet18_no_model.pt', map_location=device)
    model[0]= model[0].eval()

def resnet18_clicked(videoloop_stop, pred):
    pred[0] = True
    model[0] = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18/resnet18_model.pt', map_location=device)
    model[0]= model[0].eval()


def videoLoop(mirror=True):
    No = 0
    cap = cv2.VideoCapture(No)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)

    while True:
        ret, to_draw = cap.read()
        if mirror is True:
            to_draw = to_draw[:, ::-1]

        image = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        #if pred[0] == True:
        if True:
            image2 = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(image2)

            #Predizione modello
            image2 = transforms['test'](image2).unsqueeze(0)
            
            modello = torch.load('C:/Users/admin/Desktop/GUI_tesi/Modelli/resnet18_backup/resnet18_vecchio_model.pt', map_location=device)
            modello.eval()
            
            #modello.eval()
            pred = modello(image2.to(device))
            pred = F.softmax(pred, dim=1)
            _, class_idx[0] = torch.max(pred,1)

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
        

        
        image = ImageTk.PhotoImage(image)
        #label = tk.Label(root, text='Funziona', image=image, compound='center')
        #label.pack()
        #tk.Label(root, image=image, text="Update User",
                 #compound=tk.CENTER).pack() # Put it in the display window
        
        panel = tk.Label(root, image=image, text=class_names[class_idx[0]], compound='bottom')
        panel.config(font=("Courier", 25))
        panel.config(fg="#FFFFFF")
        panel.config(bg="green")
        panel.image = image
        panel.place(x=50, y=50)

        # check switcher value
        if videoloop_stop[0]:
            # if switcher tells to stop then we switch it again and stop videoloop
            videoloop_stop[0] = False
            panel.destroy()
            break


# videoloop_stop is a simple switcher between ON and OFF modes
videoloop_stop = [False]

pred = [False]
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
    root, text="start", bg="#fff", font=("", 20),
    command=lambda: startButton_clicked(videoloop_stop, pred))
startButton.place(x=1000, y=50, width=140, height=90)

#STOP
stopButton = tk.Button(
    root, text="stop", bg="#fff", font=("", 20),
    command=lambda: stopButton_clicked(videoloop_stop, pred))
stopButton.place(x=1160, y=50, width=140, height=90)

#RESNET18_NO
resnet18_no = tk.Button(
    root, text="resnet18_no", bg="#fff", font=("", 15),
    command=lambda: resnet18_no_clicked(videoloop_stop, pred))
resnet18_no.place(x=1000, y=250, width=140, height=80)

#RESNET18
resnet18 = tk.Button(
    root, text="resnet18", bg="#fff", font=("", 15),
    command=lambda: resnet18_clicked(videoloop_stop, pred))
resnet18.place(x=1140, y=250, width=140, height=80)

#ALEXNET
alexnet = tk.Button(
    root, text="alexnet", bg="#fff", font=("", 15),
    command=lambda: alexnet_clicked(videoloop_stop, pred))
alexnet.place(x=1000, y=330, width=140, height=80)

#GOOGLELENET
googleLeNet = tk.Button(
    root, text="googleLeNet", bg="#fff", font=("", 15),
    command=lambda: googleLeNet_clicked(videoloop_stop, pred))
googleLeNet.place(x=1140, y=330, width=140, height=80)

#MOBILENET_v2
mobilenet_v2 = tk.Button(
    root, text="mobilenet_v2", bg="#fff", font=("", 15),
    command=lambda: mobilenet_v2_clicked(videoloop_stop, pred))
mobilenet_v2.place(x=1000, y=410, width=140, height=80)

#MOBILENET_v3
mobilenet_v3 = tk.Button(
    root, text="mobilenet_v3", bg="#fff", font=("", 15),
    command=lambda: mobilenet_v3_clicked(videoloop_stop, pred))
mobilenet_v3.place(x=1140, y=410, width=140, height=80)

#efficientnet_b0
efficientnet_b0 = tk.Button(
    root, text="efficientnet_b0", bg="#fff", font=("", 15),
    command=lambda: shufflenet_v2_clicked(videoloop_stop, pred))
efficientnet_b0.place(x=1000, y=490, width=140, height=80)

#sufflenet_v2
shufflenet_v2 = tk.Button(
    root, text="shufflenet_v2", bg="#fff", font=("", 15),
    command=lambda: shufflenet_v2_clicked(videoloop_stop, pred))
shufflenet_v2.place(x=1140, y=490, width=140, height=80)



root.mainloop()
