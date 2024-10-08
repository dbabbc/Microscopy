# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:48:37 2024
@author: diego
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

#Import the image and resize it to a managable size. img1 is really f, our
#function
filename = 'nyc.jpg'
img = mpimg.imread(filename)
img1 = img[250-1:500-1,0:250,]

#Creating the figure to plot multiple images, and plotting the og and cutout
#images. Our cutout will be the one we work with. 
ogfig = plt.figure()
figrows = 2
figcols = 3

ogfig.add_subplot(figrows,figcols,1)
plt.imshow(img1)
plt.axis('off')
plt.title('f - Image Function')

#We take F as the fourier transform of the resized image f, and plot it
F = np.fft.fftshift((np.fft.fft2(img1)))

#We take the logarithm of this F to be able to see something since our og 
#values are very high and very low. We also add a very small number to 
#not have any 0s in our log
ogfig.add_subplot(figrows,figcols,4)
plt.imshow(np.log(np.abs(F)**2)+0.0000001)
plt.axis('off')
plt.title('F - FFT of image')

#We obtain the size of our image
N = img1.shape[0] 

#We bring back our gaussian function from the last excercise. We set N as its
#size, since it has to match the exact size of the cutout image to work. 
x = y = np.linspace(-1.0, 1.0, N)
X,Y = np.meshgrid(x,y)
W=0.1

#We define g as the gaussian function. This is called a kernel. Impulse response lookup 
g = np.abs(np.exp((-X**2 - Y**2)/W**2))

ogfig.add_subplot(figrows,figcols,2)
plt.imshow(np.abs(g))
plt.axis('off')
plt.title('g - Kernel')

#Take fft of g. This is called the Transfer Function. This transfer function  
#is the bandwidth of the kernel. In this case it is a very small bandwith. 
#Then we plot it.
G = np.fft.fftshift((np.fft.fft2(g)))

ogfig.add_subplot(figrows,figcols,5)
plt.imshow(np.abs(G))
plt.axis('off')
plt.title('G - Transfer Function')

#Take convolution of G and F, which is the fft of our image, and plot it
H = G*F

ogfig.add_subplot(figrows,figcols,6)
plt.imshow(np.abs(H))
plt.axis('off')
plt.title('H - Conv. F and G')

#Take the inverse FFT (ifft) of H and plot it
h = np.fft.ifftshift(np.fft.ifft2(H))

ogfig.add_subplot(figrows,figcols,3)
plt.imshow(np.abs(h))
plt.axis('off')
plt.title('h - Conv. f and g')


plt.show()