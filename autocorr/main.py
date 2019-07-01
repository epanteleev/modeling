import math
import matplotlib.pyplot as plt
from cmath import exp, pi
import numpy as np
import scipy as sig
import random

def convolution(F, G):
    lenA = len(F); lenB = len(G)
    res = [0 for i in range(lenA + lenB - 1)]
    for m in range(lenA):
        for n in range(lenB):
            res[m+n] += F[m] * G[n]
    return res


def autocorrelation(B):
    return convolution(np.conj(B), B[::-1])


def fast_conv(F,G):
    m = len(F) + len(G) - 1
    f = sig.fft(F,n=m)
    g = sig.fft(G,n=m)
    return sig.ifft([f * g])


def create_grap(discrete_g):
    f_wave = autocorrelation(discrete_g) #- np.correlate(discrete_g, discrete_g, mode='full')
    scale = [x for x in range(N)]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
    ax1.scatter(range(len(f_wave)), f_wave)
    ax1.set_title('AutoCorrelation')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')

    ax2.plot(scale, discrete_g)
    ax2.set_title('Original: G')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    plt.show()



if __name__ == "__main__":
    N = 200
    discrete_g = [random.uniform(-0.5, 0.5) for x in range(N)]
    create_grap(discrete_g)

    discrete_g = [(N // 2 + 20 > x > N // 2 - 20 if 10 else 0) for x in range(N)]

    create_grap(discrete_g)

    discrete_g = [math.sin(2 * math.pi * x / N) for x in range(N)]

    create_grap(discrete_g)

    discrete_g = [random.uniform(-0.5, 0.5) + math.sin(5 * math.pi / N * x) for x in range(N)]
    create_grap(discrete_g)


