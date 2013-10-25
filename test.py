from chirpz import chirpz
import numpy as np
from matplotlib import pyplot as plt

t = np.linspace(0, 1, 48e3)
x = np.sin(10000.*t*2*np.pi)[0:512]
x += np.sin(1000.*t*2*np.pi)[0:512]
x += np.sin(1200.*t*2*np.pi)[0:512]
x += np.sin(120.*t*2*np.pi)[0:512]

m = len(x)/4

Fs = 48e3

F1 = 0
F2 = 2000

# input, start, step, length
# def chirpz(x, A, W, M):

y = chirpz(x,
	np.e**(1j*2*np.pi*F1/Fs),
	np.e**(-1j*2.0*np.pi*(F2-F1)/(Fs*m)),
	m)

plt.semilogy(np.linspace(F1, F2, m), np.abs(y), 'or', label='czt')

plt.semilogy(np.fft.fftfreq(len(x), 1/Fs), np.abs(np.fft.fft(x)), '.k', label='fft')

plt.legend(loc='best')
plt.show()
