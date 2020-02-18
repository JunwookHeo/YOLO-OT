# YOT

## 1. Overview

This project aims to develop a neural network to track a tennis ball using YOLO version 3 and LSTM. YOLO version 3 has a key role to detect objects in an image while LSTM deals with their location in each frame as a historical data.

### 1.1 YOTMCLS

The input of YOTMCLS consists of the image feature and coordinates of an image from the YOLO output. The location of an object is predicted from LSTM.


![image](https://drive.google.com/uc?export=view&id=1umyAOEqrn5pXMiXq8tnvpKNrqHdqYFQq)


### 1.2 YOTMPMO

The coordinates are converted into probability map and then fed to LSTM. The output of LSTM is reconverted into coordinates using the below equation.


<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;C_x&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;C_y&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;W&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;-&space;C_x)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;H&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;-&space;C_y)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;C_x&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;C_y&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;W&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;-&space;C_x)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;H&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;-&space;C_y)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\" title="\\ C_x = \frac{1}{n} \sum^{n-1}_{0} \sum^{n-1}_{0} (x_{ij} + 0.5) \ \ \ \ if \ P_{ij} >= 0.5 \\ C_y = \frac{1}{n} \sum^{n-1}_{0} \sum^{n-1}_{0} (y_{ij} + 0.5) \ \ \ \ if \ P_{ij} >= 0.5 \\ W = 0.5 + k \sqrt{ \sum^{n-1}_{0} \sum^{n-1}_{0} (x_{ij} - C_x)^2} \ \ \ \ if \ P_{ij} >= 0.5 \\ H = 0.5 + k \sqrt{ \sum^{n-1}_{0} \sum^{n-1}_{0} (y_{ij} - C_y)^2} \ \ \ \ if \ P_{ij} >= 0.5 \\" /></a>

![image](https://drive.google.com/uc?export=view&id=1ayQWXQ9VW5zZDVgEB9ebelCUtu1LnKqR)



### 1.3 YOTMMLP

LSTM for each coordinate is separated in the YOTMMLP model, so Cx, Cy, W, H are independently predicted.

![image](https://drive.google.com/uc?export=view&id=1fnkc93bKByZNVaBa5_RmuAPXN83wRnD7)



## 2. Prerequisites

Python 3.7  
PyTorch 1.3  

## 3. Test Results


### 3.1 YOTMCLS

### 3.2 YOTMPMO
This model shows poor performance with overfitting.

![image](https://drive.google.com/uc?export=view&id=1sulOkpk7Y226fUaJuwtd1iUX7ryOQuOo)

### 3.3 YOTMMLP
With 64 of hidden size, YOTMMLP shows good performance.

![image](https://drive.google.com/uc?export=view&id=1LpP4XcTehhcv-mqvGka6HgErITCdMr0u)


### 3.3 YOTMMLP with GT

Ground truth is also sequential data, so training with ground truth and YOLO output will be expected to improve the performance. The result is above. In this case, the hidden size of LSTM is 32 and this model shows slightly better performance than YOTMMLP trained without ground truth.


![image](https://drive.google.com/uc?export=view&id=1NqQSlK-rdVeJ6XCkt2oslmzPK6W_bo7n)


## References

https://github.com/Guanghan/ROLO  

https://github.com/eriklindernoren/PyTorch-YOLOv3  



