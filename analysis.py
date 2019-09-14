import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def show_image_from_tensor():
    # pytorch provides a function to convert PIL images to tensors.
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Read the image from file. Assuming it is in the same directory.
    pil_image = Image.open('imgs/dog.jpg')
    rgb_image = pil2tensor(pil_image)
    print(rgb_image.shape)

    # Plot the image here using matplotlib.
    def plot_image(tensor):
        plt.figure()
        # imshow needs a numpy array with the channel dimension
        # as the the last dimension so we have to transpose things.
        plt.imshow(tensor.numpy().transpose(1, 2, 0))
        plt.show()

    plot_image(rgb_image)

    # Show the image tensor type and tensor size here.
    print('Image type: ' + str(rgb_image.type()))
    print('Image size: ' + str(rgb_image.size()))

#show_image_from_tensor()

def torch_max():
	h = torch.randn(2,5, requires_grad=True)
	print(h)
	#val,idx = h.max(1, keepdim=True)
	val,idx = torch.max(h, 1)
	print(val)
	print(idx)

#torch_max()


def torch_min_max():
	h1 = torch.randn(1, requires_grad=True)
	h2 = torch.randn(3, requires_grad=True)
	print(h1)
	print(h2)
	#val,idx = h.max(1, keepdim=True)
	val = torch.max(h1, h2)
	print(val)
	val = torch.min(h1, h2)
	print(val)

#torch_min_max()

import torch.nn as nn
import torch.nn.functional as F
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
 
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

def test_upsample():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	x = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
	#model = Upsample(2)
	model = Upsample(scale_factor = 4, mode = "nearest")
	p = model(x)
	print(x.shape)
	print(p.shape)
	print(x)
	print(p)

#test_upsample()

def test_rectangle():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    X, Y = 416, 416
    plt.figure()
    ax = plt.gca()
    rect = patches.Rectangle((100, 100 ), 100, 100, edgecolor='r', fill=None)
    ax.add_patch(rect)
    plt.ylim((Y, 0))
    plt.xlim((0, X))
    plt.show()

#test_rectangle()

def draw_anchor():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # anchors = [(10,13), (16,30), (33, 23)]
    # G = (512/52, 512/52)

    anchors = [(30,61), (62,45), (59, 119)]
    G = (512/26, 512/26)

    # anchors = [(116,90), (156,198), (373, 326)]
    # G = (512/13, 512/13)

    N = 7
    plt.figure()
    ax = plt.gca()
    X, Y = G
    X *= N
    Y *= N
    C = (X/2, Y/2)

    for i in range(1, N):
        ax.add_line(plt.Line2D([0, X], [G[1]*i, G[1]*i], color='black'))
        ax.add_line(plt.Line2D([G[1]*i, G[1]*i], [0, Y], color='black'))
    
    for anc in anchors:    
        rect = patches.Rectangle((C[0]-anc[0]/2, C[1]-anc[1]/2 ), anc[0], anc[1], edgecolor='r', linewidth=3, fill=None)
        ax.add_patch(rect)

    plt.ylim((Y, 0))
    plt.xlim((0, X))
    plt.show()

draw_anchor()
