# YOT

## 1. Overview

This project aims to develop a neural network to track an object using YOLO V3 and LSTM. YOLO V3 has a key role to detect objects in an image while LSTM deals with their location in each frame as a historical data. In this project, three models are trained and estimated. YOTMCLS is utilizing coordinates and image feature from YOLO V3 as an input data. YOTMPMO does not use image feature data. In this model, coordinates are converted into probability map as an input data. YOTMMLP is designed that Cx, Cy, W and H are fed to its own LSTM network separately.

### 1.1 YOTMCLS

The input of YOTMCLS consists of the image feature and coordinates of an image from the YOLO output. The location of an object is predicted from LSTM.


![YOTMCLS](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/YOTMCLS.png)


### 1.2 YOTMPMO

The coordinates are converted into probability map and then fed to LSTM. However, there is no way to reconvert the probability map into coordinates. Thus, this project proposes the way to convert the output of LSTM into coordinates using the below equation.


<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;C_x&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;C_y&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;W&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;-&space;C_x)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;H&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;-&space;C_y)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;C_x&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;C_y&space;=&space;\frac{1}{n}&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;&plus;&space;0.5)&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;W&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(x_{ij}&space;-&space;C_x)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\&space;H&space;=&space;0.5&space;&plus;&space;k&space;\sqrt{&space;\sum^{n-1}_{0}&space;\sum^{n-1}_{0}&space;(y_{ij}&space;-&space;C_y)^2}&space;\&space;\&space;\&space;\&space;if&space;\&space;P_{ij}&space;>=&space;0.5&space;\\" title="\\ C_x = \frac{1}{n} \sum^{n-1}_{0} \sum^{n-1}_{0} (x_{ij} + 0.5) \ \ \ \ if \ P_{ij} >= 0.5 \\ C_y = \frac{1}{n} \sum^{n-1}_{0} \sum^{n-1}_{0} (y_{ij} + 0.5) \ \ \ \ if \ P_{ij} >= 0.5 \\ W = 0.5 + k \sqrt{ \sum^{n-1}_{0} \sum^{n-1}_{0} (x_{ij} - C_x)^2} \ \ \ \ if \ P_{ij} >= 0.5 \\ H = 0.5 + k \sqrt{ \sum^{n-1}_{0} \sum^{n-1}_{0} (y_{ij} - C_y)^2} \ \ \ \ if \ P_{ij} >= 0.5 \\" /></a>

![YOTMPMO](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/YOTMPMO.png)



### 1.3 YOTMMLP

There are four LSTMs for each coordinate in the YOTMMLP model, so Cx, Cy, W, H are independently predicted. Because separating coordinates makes the prediction model simple, it is expected that the performance will improve.

![YOTMMLP](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/YOTMMLP.png)



## 2. Prerequisites

Python 3.7  
PyTorch 1.3  

## 3. Dataset and Training

### 3.1 Dataset

To train YOT, 27 of TB-100 data from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html are used.

### 3.2 Training

60% frames of each video clip are used to train the networks and 20% frames of them are used to validate them. The IoT scores of training and validating sets of YOLO outputs are 0.641 and 0.646.

### 3.3 Default value of coordinates

The default value of coordinates of a predicted object from YOLO V3 is (0, 0, 0, 0, 0) when the object is not detected. However, using Cx=0 and Cy=0 may make a bias because (0, 0) means left-top in an image. In this project, (0.5, 0.5, 0, 0, 0) is used as the default value for undetected object.


## 4. Test Results

### 4.1 YOTMCLS

### 4.2 YOTMPMO
This model shows poor performance with overfitting.

![YOTMPMO](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/Result_YOTMPMO.png)



### 4.3 YOTMMLP
With 64 of hidden size, YOTMMLP shows good performance.

![YOTMMLP](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/Result_YOTMMLP.png)

Demo videos are available. 

[Video 1](https://youtu.be/uM0QMNIsCD8),
[Video 2](https://youtu.be/Q2IWIX3TtiA)


### 4.3 YOTMMLP with GT

Ground truth is also sequential data, so training with ground truth and YOLO output will be expected to improve the performance. The result is above. In this case, the hidden size of LSTM is 32 and this model shows slightly better performance than YOTMMLP trained without ground truth.


![YOTMMLP with GT](https://github.com/JunwookHeo/YOLO-OT/blob/master/Report/Result_YOTMMLP_with_GT.png)


## References

https://github.com/Guanghan/ROLO  

https://github.com/eriklindernoren/PyTorch-YOLOv3  



