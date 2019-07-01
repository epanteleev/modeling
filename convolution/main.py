import math
import matplotlib.pyplot as plt
import numpy as np
from cmath import exp, pi
import scipy as sig
import random
# фильтр Баттер-Вотта

def convolution(F, G):
    lenA = len(F); lenB = len(G)
    res = [0 for i in range(lenA + lenB - 1)]
    for m in range(lenA):
        for n in range(lenB):
            res[m+n] += F[m] * G[n]
    return res


def fast_conv(F,G):
    m = len(F) + len(G) - 1
    f = sig.fft(F,n=m)
    g = sig.fft(G,n=m)
    return sig.ifft([f * g])


def autocorrelation(A, B):
    return convolution(np.conj(A), B[::-1])


def create_graphic(discrete_f, discrete_g):
    f_wave = autocorrelation(discrete_f, discrete_g)
    d_wave = convolution(discrete_f, discrete_g)
    scale = [x for x in range(len(discrete_f) + len(discrete_g) - 1)]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10, 8))
    ax1.scatter(scale, d_wave)
    ax1.set_title('Convolution')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    ax2.plot(scale[0:len(discrete_f)], discrete_f)
    ax2.set_title('Original: F')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    ax3.plot(scale[0:len(discrete_g)], discrete_g)
    ax3.set_title('Original: G')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')

    ax4.scatter(scale[:len(f_wave)], f_wave  , marker='o', c='r', edgecolor='b')
    ax4.set_title('Autocorr:')
    ax4.set_xlabel('$x$')
    ax4.set_ylabel('$y$')
    plt.show()


if __name__ == "__main__":
    N = 500
    M = 800
    discrete_f = [(N // 2 + 20 > x > N // 2 - 20 if 10 else 0) for x in range(N)]
    discrete_g = [(M // 2 + 20 > x > M // 2 - 20 if 10 else 0) for x in range(M)]

    create_graphic(discrete_f, discrete_g)

    discrete_f = [math.sin(6 * math.pi/N*x) for x in range(N)]
    create_graphic(discrete_f, discrete_g)
    
    discrete_g = [random.uniform(-0.5,0.5) for x in range(N)]
    discrete_f = [random.uniform(-0.5,0.5) for x in range(M)]
    create_graphic(discrete_f, discrete_g)

