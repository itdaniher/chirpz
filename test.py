from chirpz import chirpz
import numpy as np
import matplotlib

#1matplotlib.use('cairo')
from matplotlib import pyplot as plt

t = np.linspace(0, 1, 48e3)
freqs = [120, 1000, 1200, 2200]
x = np.random.rand(512) - 0.5
x *= 5
for freq in freqs:
    x += np.sin(freq*t[:512]*2*np.pi)

x /= np.mean(x)

plt.title('input vector')
plt.plot(x)
plt.xlabel('samples')
plt.figure()

print('actual frequency content', freqs)
m = len(x)//2

Fs = 48e3
F1 = 900
F2 = 3000

# input, start, step, length
# def chirpz(x, A, W, M):
A = np.e**(1j*2*np.pi*F1/Fs)
W = np.e**(-1j*2.0*np.pi*(F2-F1)/(Fs*m))

cztAmps = np.abs(
		chirpz(x,
		A,
		W,
		m)
	)

cztFreqs = np.linspace(F1, F2, m)

cztPeaks = [cztFreqs[i-1] for i in range(m) if cztAmps[i-2] < cztAmps[i-1] > cztAmps[i] if cztAmps[i] > max(cztAmps)*0.5]
print('freq peaks from czt / zoom FFT', cztPeaks)

plt.vlines(cztPeaks, 0, max(cztAmps), 'r', label='peaks from czt')

plt.plot(cztFreqs, cztAmps, 'or', label='czt')

fftFreqs = np.linspace(0, Fs/2, len(x)*2)

fftAmps = np.abs(np.fft.rfft(x, len(x)*4))[1::]

fftPeaks = [fftFreqs[i-1] for i in range(m) if fftAmps[i-2] < fftAmps[i-1] > fftAmps[i] if fftAmps[i-1] > max(fftAmps)*0.5]
print('freq peaks from fft', fftPeaks)
plt.vlines(fftPeaks, 0, max(fftAmps), 'k', label='peaks from fft')

plt.plot(fftFreqs, fftAmps, '.k', label='fft')

plt.xlabel('freq (Hz)')
plt.ylabel('amplitude')
plt.title('frequency domain transform comparisons for high precision signal identification')

plt.legend(loc='best')
plt.show()
