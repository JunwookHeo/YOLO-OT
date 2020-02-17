# YOT

## Overview

This project aims to develop a neural network to track a tennis ball using YOLO version 3 and LSTM. YOLO version 3 has a key role to detect objects in an image while LSTM deals with their location in each frame as a historical data.

### YOTMCLS

The input of YOTMCLS consists of the image feature and coordinates of an image from the YOLO output. The location of an object is predicted from LSTM.


![image](https://drive.google.com/uc?export=view&id=1umyAOEqrn5pXMiXq8tnvpKNrqHdqYFQq)


### YOTMPMO

The coordinates are converted into probability map and then fed to LSTM. The output of LSTM is reconverted into coordinates.

![image](https://drive.google.com/uc?export=view&id=1ayQWXQ9VW5zZDVgEB9ebelCUtu1LnKqR)



### YOTMMLP

LSTM for each coordinate is seperated in the YOTMMLP model, so Cx, Cy, W, H are independently predicted.

![image](https://drive.google.com/uc?export=view&id=1fnkc93bKByZNVaBa5_RmuAPXN83wRnD7)



## Prerequisites

Python 3.7  
PyTorch 1.3  

## References

https://github.com/Guanghan/ROLO  

https://github.com/eriklindernoren/PyTorch-YOLOv3  



