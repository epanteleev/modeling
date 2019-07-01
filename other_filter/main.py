# фильтр скользящего среднего
# фильтр с прямоугольной ачх

from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq, irfft
from numpy.random import uniform
from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import scipy

FD = 22050
N = 2000


def BatterLaw(X, Wc, n):
    Hw = np.zeros(N)
    for i in range(len(X)):
        k = 1 / (1 + (X[i] / Wc) ** (2 * n))
        Hw[i] = k
        if (k < 0.01):
            print("returned")
            return Hw

    return Hw


def BatterHight(X, Wc, n):
    Hw = np.ones(len(X))
    for i in range(len(X)):
        if (X[i] != 0):
            Hw[i] = (1 / (1 + (Wc / X[i]) ** (2 * n)))
            if (1 - Hw[i] < 0.01):
                return Hw
        else:
            Hw[i] = 0
    return Hw


def BatterBand(X, WL, WH, n):
    hightFilter = BatterHight(X, WH, n)
    lowFilter = BatterLaw(X, WL, n)
    resFilter = hightFilter * lowFilter
    return resFilter


if __name__ == '__main__':
    sig = [6. * sin(2. * pi * 440.0 * t / FD) + 2 * sin(pi * 4400.0 * t / FD) for t in range(N)]

    spectrum = rfft(sig)
    sigFft = scipy.fft(sig)

    freq = rfftfreq(N, 1. / FD)

    # batterLaw = BatterLaw(freq, 1000, 10)
    batterHight = BatterHight(freq, 1000, 10)

    batterFourier = irfft(batterHight)

    newFilter = np.zeros(2 * len(spectrum))
    # newFilter = np.full(2*len(spectrum), 1)
    for i in range(100):
        newFilter[i] = batterFourier[i]

    newFilter = batterHight
    newFilter = rfft(newFilter)

    filtered = spectrum[:len(newFilter)] * newFilter

    # res = rfft(filtered)

    # filtered = spectrum - filtered
    res = rfft(filtered)
    # r = res.max()
    # print(np_abs(r))
    # res = sig[:len(res)] - res

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    fig, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    fig, (ax5, ax6) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    # fig, (ax7, ax8) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    # fig, (ax9, ax10) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    # fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    ax1.plot(range(len(sig)), sig, 'b')
    ax1.set_title('Signal')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.plot(freq, np_abs(spectrum), 'b')
    ax2.set_title('spec')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # ax7.plot(, np_abs(diff), 'b')
    # ax7.set_title('filtered')
    # ax7.set_xlabel('X')
    # ax7.set_ylabel('Y')

    # new_spec = fourie_filter(spectrum, freq, 600)

    ax3.plot(freq[:len(filtered)], np_abs(filtered), 'b')
    ax3.set_title('filtered spectrum')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    ax4.plot(arange(len(res)), res, 'b')
    ax4.set_title('result')

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    ax5.plot(range(len(batterFourier)), batterFourier, 'b')
    ax5.set_title('Fourier height')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    ax6.plot(freq[:len(newFilter)], newFilter, 'b')
    ax6.set_title('new AFS low')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')

    # ax7.plot(range(len(resHight)), resHight, 'b')
    # ax7.set_title('filtered height')
    # ax7.set_xlabel('X')
    # ax7.set_ylabel('Y')

    # ax8.plot(freq[:len(filtered)], np_abs(filteredHight) / N, 'b')
    # ax8.set_title('spectrum hight')
    # ax8.set_xlabel('X')
    # ax8.set_ylabel('Y')

    # T = rfftfreq(N)
    # sig2 = np.asarray([6. * sin(t*100) + 2*sin(t*2000) + sin(t*300) +6. * sin(t*15) +  + 3*sin(t*3000) for t in T])

    # spectrum2 = rfft(sig2)

    # freq2 = rfftfreq(N, 1. / FD)

    # batterBand = BatterBand(freq2, 200, 800, 10)

    # filtered2 = spectrum2*batterBand[:len(spectrum2)]
    # res = rfft(filtered2)

    # ax9.plot(T, sig2[:len(T)], 'b')
    # ax9.set_title('sig2')
    # ax9.set_xlabel('X')
    # ax9.set_ylabel('Y')

    # ax10.plot(range(len(res)), res, 'b')
    # ax10.set_title('result')
    # ax10.set_xlabel('X')
    # ax10.set_ylabel('Y')

    plt.show()