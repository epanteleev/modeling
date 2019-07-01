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


def BatterworthLaw(X, Wc, n):
    freq = rfftfreq(N, 1. / FD)
    Hw = [ complex(i, 0) for i in range(N)]
    for i in range(len(X)):
        Hw[i] = complex(i, 1 / (1 + (X[i] / Wc) ** (2 * n)))
        if(Hw[i].imag == 0.0):
            break
    for i in range(len(X)):
        Hw[len(X) - i] = Hw[i]
        if (Hw[len(X) - i].imag == 0.0):
            break
    # for i in range(len(X)):
    #     Hw[len(X) - i] = complex(1 / (1 + (X[i] / Wc) ** (2 * n)), freq[i])
    #     if(Hw[i] == complex(0.0, 0.0)):
    #         break
    return [complex(freq[i], Hw[i].imag) for i in range(len(X))]
    return Hw


def BatterworthHight(X, Wc, n):
    Hw = np.ones(len(X))
    for i in range(len(X)):
        if (X[i] != 0):
            Hw[i] = (1 / (1 + (Wc / X[i]) ** (2 * n)))
        else:
            Hw[i] = 0

    return Hw


def BatterworthBand(X, WL, WH, n):
    hightFilter = BatterworthHight(X, WH, n)
    lowFilter = BatterworthLaw(X, WL, n)
    return hightFilter * lowFilter


def BatterworthLaw_(X, Wc, n):
    Hw = np.ones(len(X))
    res = BatterworthHight(X, Wc, n)
    return [Hw[i] - res[i] < 0.4 if 0 else Hw[i] - res[i] for i in range(len(X))]


if __name__ == '__main__':
    sig = [6. * sin(2. * pi * 440.0 * t / FD) + 2 * sin(pi * 4400.0 * t / FD) for t in range(N)]

    spectrum = rfft(sig)
    sigFft = scipy.fft(sig)

    freq = rfftfreq(N, 1. / FD)

    batterLaw = BatterworthLaw(freq, 1000, 20)
    batterHight = BatterworthHight(freq, 1000, 20)

    butterworth = np.fft.ifft(batterHight)
    butterworth2 = np.fft.ifft(batterLaw)

    newButterworth = np.zeros(2 * len(spectrum))
    newButterworth2 = np.zeros(2 * len(spectrum))
    for i in range(1000):
        newButterworth[i] = butterworth[i]
        newButterworth2[i] = butterworth2[i]

    newButterworth = np.fft.fft(newButterworth)
    newButterworth2 = np.fft.fft(newButterworth2)

    filtered2 = spectrum * newButterworth2[:len(spectrum)]
    filtered = spectrum * newButterworth[:len(spectrum)]

    # res = rfft(filtered)

    # filtered = spectrum - filtered
    res = rfft(filtered)
    res2 = irfft(filtered2)

    # r = res.max()
    # print(np_abs(r))
    res2 = sig - res2

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    fig, (ax3, ax4, ax8) = plt.subplots(nrows=3, ncols=1, figsize=(16, 10))

    fig, (ax5, ax6, ax7) = plt.subplots(nrows=3, ncols=1, figsize=(16, 10))
    #fig, (ax7, ax8) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    fig, (ax9, ax10) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    # fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    ax9.plot(freq, [batterLaw[i].imag for i in range(len(freq))])
    ax10.plot(freq, batterHight[:len(freq)])
    #ax2.plot(freq, np.abs(spectrum[:len(freq)]) / N, 'b')
    ax1.plot(range(len(sig)), sig, 'b')
    ax1.set_title('Signal')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.plot(freq, np_abs(spectrum), 'b')
    ax2.set_title('spec')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ###
    ax3.plot(freq[:len(filtered)], np_abs(filtered), 'b')
    ax3.set_title('filtered spectrum')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    ax4.plot(arange(len(res)), res, 'b')
    ax4.set_title('result')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    ax8.plot(arange(len(res2)), res2, 'b')
    ax8.set_title('result 2')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ##
    ax5.plot(range(len(butterworth2)), butterworth2, 'b')
    #ax5.set_title('Fourier height')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    ax6.plot(freq, newButterworth[:len(freq)], 'b')
    ax6.set_title('new AFS low')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')

    ax7.plot(freq, newButterworth2[:len(freq)], 'b')
    ax7.set_title('filtered height')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')

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