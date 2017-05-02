"""Chirp z-Transform.

As described in

Rabiner, L.R., R.W. Schafer and C.M. Rader.
The Chirp z-Transform Algorithm.
IEEE Transactions on Audio and Electroacoustics, AU-17(2):86--92, 1969

http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/015_czt.pdf
"""

import numpy as np

def chirpz(x, A, W, M):

    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}
    """

    """ DFT:
    A = 1 (radius)
    a = cmath.exp(2j * pi * f1/Fs)
    M = N (len output = len input)
    W = e ^ (-j*tau/N) (curve spans 0 to 2pi)
    w = cmath.exp(-2j * pi * (f2-f1) / ((m-1)*Fs))
    """

    x, A, W, M = np.asarray(x, dtype=np.complex), np.complex(A), np.complex(W), int(M)
    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))
    n = np.arange(N,dtype=float)
    # y = A ^ (-n) * W ^ (n**2 / 2) * x
    y = np.power(A,-n) * np.power(W,n**2 / 2.) * x 
    v = np.zeros(L,dtype=np.complex)
    v[:M] = np.power(W,-n[:M]**2/2.)
    v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
    # fast convolve, undo, select first M samples
    g = np.fft.ifft(np.fft.fft(v) * np.fft.fft(y, L))[:M]
    k = np.arange(M)
    g *= np.power(W,k**2 / 2.)
    return g
