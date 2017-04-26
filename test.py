from chirpz import chirpz
import numpy as np
import matplotlib

matplotlib.use('qt5agg')
from matplotlib import pyplot as plt

t = np.linspace(0, 1, 48e3)
freqs = [2200, 1200, 1000, 120]
x = np.random.rand(512)/100
for freq in freqs:
    x += np.sin(freq*t*2*np.pi)[0:512]

print(freqs)
m = len(x)//4

Fs = 48e3

F1 = 0
F2 = 3000

# input, start, step, length
# def chirpz(x, A, W, M):

cztAmps = np.abs(
		chirpz(x,
		np.e**(1j*2*np.pi*F1/Fs),
		np.e**(-1j*2.0*np.pi*(F2-F1)/(Fs*m)),
		m)
	)

cztFreqs = np.linspace(F1, F2, m)

cztPeaks = [cztFreqs[i-1] for i in range(m) if cztAmps[i-2] < cztAmps[i-1] > cztAmps[i] if cztAmps[i] > max(cztAmps)*0.5]
print(cztPeaks)

plt.vlines(cztPeaks, 0, max(cztAmps), 'r', label='peaks from czt')

plt.plot(cztFreqs, cztAmps, 'or', label='czt')

fftFreqs = np.linspace(0, Fs/2, len(x)/2)

fftAmps = np.abs(np.fft.rfft(x))[1::]

fftPeaks = [fftFreqs[i-1] for i in range(m) if fftAmps[i-2] < fftAmps[i-1] > fftAmps[i] if fftAmps[i-1] > max(fftAmps)*0.5]
print(fftPeaks)
plt.vlines(fftPeaks, 0, max(fftAmps), 'k', label='peaks from fft')

plt.plot(fftFreqs, fftAmps, '.k', label='fft')

plt.xlabel('freq (Hz)')
plt.ylabel('amplitude')
plt.title('frequency domain transform comparisons for high precision signal identification')

plt.legend(loc='best')
plt.show()
