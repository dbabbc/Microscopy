# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:53:22 2024

@author: diego
"""

#Preguntas: Que es el operador "|", Que hace poner algo en brackets [],
#en la def function,

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy import fft


#Framework for figures
figs = plt.figure()
figrows = 2
figcols = 3

#Defining the function of the free space propagator
def D(x,y,z,wavelength,n):
    
    k = n/wavelength
    r = np.sqrt(x**2+y**2+z**2)
                
    return - 1.j*k*z*(1+1.j/(2*pi*k*r))*np.exp(1.j*2*pi*k*r)/r**2 

#Defining a micron
um = 1.0

#Defining the length of the space
L = 300*um

#Creating the space which is 150um**2 with 1024 # of points 
x = np.linspace(-L/2,L/2,1024)
y = np.linspace(-L/2,L/2,1024)

X,Y = np.meshgrid(x,y)

#Defining the electric field E0 which is a plane wave
E0 = np.ones([1024,1024])

#If we wanted the phase to be at an angle, we need to
#introduce a phase term.This term is the same wave, but at an angle ky
kx = 0
ky = 0.1
E01 = np.exp(-1.j*2*pi*(ky*Y+kx*X))

#We can make it a cosine, which gives a sinusoidal propagation
#of the field.
E02 = np.cos(2*pi*(ky*Y+kx*X))

#Putting the mask on the space. Inside
#the mask everything is 1, outside it everything is 0. 
side = 30*um
indexes = (np.abs(X)>side/2) | (np.abs(Y)>side/2)
E0[indexes]=0
E01[indexes]=0
E02[indexes]=0

#Plotting the Electric Field with a mask in the center. Making a suplot func.
#and adjusting the spacing between plots with the adjust function. 

figs.add_subplot(figrows,figcols,1)
figs.subplots_adjust(wspace=0.5, hspace=0.25)
plt.imshow(E0)
plt.title('Plane Wave') 

figs.add_subplot(figrows,figcols,2)
plt.imshow(np.abs(E01))
plt.title('Phase at an angle')

figs.add_subplot(figrows,figcols,3)
plt.imshow(np.abs(E02))
plt.title('Sine Phase')

#We want to get the convolution of D and E0. We call the function
#D and input our values, naming the return Dz. D at the distance Z
Dz =  D(X,Y,1500*um,0.4*um,1)

#Now we take the fourier transforms of both D and E0
ft_Dz = fft.fftshift(fft.fft2(Dz))
ft_E0 = fft.fftshift(fft.fft2(E0))
ft_E01 = fft.fftshift(fft.fft2(E01))
ft_E02 = fft.fftshift(fft.fft2(E02))

#Now we get the fft of the field E we want to get in the end.
ft_E = ft_Dz*ft_E0
ft_E1 = ft_Dz*ft_E01
ft_E2 = ft_Dz*ft_E02

#Getting the electric field E from the inverse fft of 
# the convolution of D*E0
E = fft.ifftshift(fft.ifft2(ft_E))
E1 = fft.ifftshift(fft.ifft2(ft_E1))
E2 = fft.ifftshift(fft.ifft2(ft_E2))

figs.add_subplot(figrows,figcols,4)
plt.imshow(np.abs(E))


figs.add_subplot(figrows,figcols,5)
plt.imshow(np.abs(E1))
plt.title('Propagation of E')

figs.add_subplot(figrows,figcols,6)
plt.imshow(np.abs(E2))

