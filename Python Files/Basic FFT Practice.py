import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt

"Example of simple fourier transform"

x = y = np.linspace(-1.0, 1.0, 11)
X,Y = np.meshgrid(x,y)

W=0.5

Z = np.abs(np.exp((-X**2 - Y**2)/W**2))

FZ = fftshift(fft2(Z))

plt.figure()
plt.imshow(Z)
plt.show()

plt.figure()
plt.imshow(np.abs(FZ))
plt.show()
