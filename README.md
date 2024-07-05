# Traffic Sign Recognition

This project is designed to recognize and classify traffic signs using various pre-trained deep learning models. The models used in this project include ResNet18, AlexNet, GoogleLeNet, MobileNet_v2, MobileNet_v3, EfficientNet_b0, and ShuffleNet_v2. The dataset used for training and testing is the German Traffic Sign Recognition Benchmark (GTSRB).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Performance](#models-and-performance)
- [Evaluation](#evaluation)
- [GUI](#gui)
- [Acknowledgments](#acknowledgments)

## Introduction

The goal of this project is to create a robust traffic sign recognition system. It leverages several state-of-the-art convolutional neural network (CNN) architectures and evaluates their performance based on accuracy and frames per second (FPS) on both GPU and CPU.
![Pepper at Hotel Reception](https://miro.medium.com/v2/resize:fit:1400/1*e0UlsRVfTM2xw_uVWTsPVg.png)

## Dataset

The dataset used in this project is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip). 

To download the dataset:

```bash
!wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
!unzip -qq GTSRB_Final_Training_Images.zip
```

## Installation

To set up the environment for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and unzip it in the project directory.

## Usage

1. To train the models, run the Jupyter notebook `Tesi_Traffic_Sign_Recognition.ipynb`.

2. Use the provided Python scripts to evaluate different models and visualize the results.

3. To use the GUI for real-time traffic sign recognition, run:
   ```bash
   python gui.py
   ```

## Models and Performance

The table below summarizes the performance of different models:

| Model             | Parameters    | Accuracy | avgFPS (GPU) | avgFPS (CPU) |
|-------------------|---------------|----------|--------------|--------------|
| Resnet18_no_pretrain | 11.4 million | 96.516   | 268.195      | 7            |
| Resnet18          | 11.4 million  | 99.074   | 268.132      | 7            |
| Alexnet           | 62.3 million  | 97.878   | 778.899      | 16           |
| GoogleLeNet       | 6.7 million   | 99.192   | 100.585      | 5            |
| MobileNet_v2      | 3.5 million   | 98.907   | 127.573      | 9            |
| MobileNet_v3      | 2.54 million  | 98.741   | 135.301      | 25           |
| Efficientnet_b0   | 5.29 million  | 99.256   | 79.924       | 5            |
| Shufflenet_v2     | 2.29 million  | 76.888   | 109.488      | 14           |

## Evaluation

To evaluate the models, use the test dataset provided in the GTSRB. The performance is measured based on accuracy and FPS. Use the following command to start the evaluation process:

```python
python evaluate.py
```

## GUI

A graphical user interface (GUI) is provided to facilitate real-time traffic sign recognition using a webcam. The GUI allows users to select different models and see their predictions in real-time.

To launch the GUI, run:

```bash
python gui.py
```

# Demo video
Link video funzionamento dell'interfaccia per l'uso dei modelli addestrati in real-time con webcam pc: https://youtu.be/jAFBdIOcRgY

