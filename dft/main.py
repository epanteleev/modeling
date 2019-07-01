import math
import numpy as np
import matplotlib.pyplot as plt
from cmath import exp, pi
import random
import scipy.fftpack as sci

N = 1000


def dft(xs):
    n = len(xs)
    return [sum((xs[k] * exp(-2j * math.pi * i * k / n) for k in range(n))) for i in range(n)]


def dft_(xs):
    n = len(xs)
    res = []
    for k in range(n):
        for i in range():

        res

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]


if __name__ == "__main__":
    A = 2
    frequency = 1.5
    win = [1 for x in range(N)]

    wave = [win[x] * A * math.sin(frequency * math.pi/N*x) for x in range(N//2)]
    for x in range(N//2):
        wave.append(win[x+ N//2]*A * math.sin(frequency * math.pi/N*(x)))

    rng = [frequency * math.pi/N*x for x in range(N)]
   # wave = win
    #f_wave = sci.fft(wave)
    f_wave = dft(wave)
    #f_wave = sci.dct(wave)
    range_wave = [sci.fft(wave) - f_wave]

    re_wave = [np.abs(x) for x in f_wave]
    im_wave = [np.angle(x) for x in f_wave]
#    r_im_wave = [(y / x) for x in imf_wave for y in rf_wave ]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,figsize=(10, 8))

    ax1.plot(rng[:50], re_wave[:50])
    ax1.set_title('Fourier Transform')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    ax2.plot(rng, wave)
    ax2.set_title('Original')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    ax3.scatter(rng, range_wave)
    ax3.set_title('Range')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    plt.show()