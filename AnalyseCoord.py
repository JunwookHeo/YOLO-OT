import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tkinter import filedialog, Tk
#from tkinter import *

import glob

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