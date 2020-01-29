import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tkinter import filedialog, Tk
#from tkinter import *

import glob

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_iou(box1, box2):
	# determine the (x, y)-coordinates of the intersection rectangle
    b1_x1, b1_x2 = box1[0] - box1[2]/2, box1[0] + box1[2]/2
    b1_y1, b1_y2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
    b2_x1, b2_x2 = box2[0] - box2[2]/2, box2[0] + box2[2]/2
    b2_y1, b2_y2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2

    ix1 = np.maximum(b1_x1, b2_x1)
    iy1 = np.maximum(b1_y1, b2_y1)
    ix2 = np.minimum(b1_x2, b2_x2)
    iy2 = np.minimum(b1_y2, b2_y2)

    # compute the area of intersection rectangle
    interArea = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box1Area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    box2Area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (box1Area + box2Area - interArea + 1e-16)

    # return the intersection over union value
    return iou

def parse_coord(path):
    try:
        coords = np.load(path)
    except:
        print('Cannot open the file....')
    
    name = os.path.basename(path)
    coords = coords.transpose((1,2,0))
    cp = coords[0]
    cy = coords[1]
    ct = coords[2]

    titles = ['cx', 'cy', 'width', 'height']
    fig = figure()
    for i, (cpi, cyi, cti) in enumerate(zip(cp, cy, ct)):
        axs = plt.subplot(4, 1, i+1)
        plt.plot(cpi)
        plt.plot(cyi)
        plt.plot(cti)
        axs.set_ylabel(titles[i])
    
    yiou = bb_iou(cy, ct)
    piou = bb_iou(cp, ct)
    print(f'{name} - Avg IOU : YOT={np.mean(piou)}, YOLO={np.mean(yiou)}')
    fig.legend(labels=['predict', 'yolo', 'ground th'], loc='right')
    fig.suptitle(name)
    plt.show()

    

def main(argvs):
    root = Tk()
    #root.filename = filedialog.askopenfilename(initialdir = './', title = 'Select file', filetypes = (('Numpy files', '*.npy'), ('all files', '*.*')))
    #parse_coord(root.filename)

    path = filedialog.askdirectory()
    print(path)
    if path:
        files = glob.glob("%s/*.npy" % path)
        for f in files:
            parse_coord(f)

if __name__=='__main__':
    main('')
    