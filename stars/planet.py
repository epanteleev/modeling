from numpy.fft import rfft, rfftfreq, irfft, fft, ifft, fftfreq
import numpy.fft
from math import sin, pi
import matplotlib.pyplot as plt
import numpy as np
import cmath

N = 3000
FD = 30000

def BandButterWorthFilter(sig, left, right):
    freq_axis = fftfreq(N, 1.0 / FD)
    time_axis = np.arange(N)
    filter = np.ones(len(freq_axis), dtype=complex)
    border = left + right
    phase = 0
    print(freq_axis[left], " -- ", freq_axis[right])
    # for i in range(2*(int(N/2) - border )):
    #     filter[i + int(N/2) - (int(N/2) - border)] = 1

    for i in range(left-100, right+100):
        filter[i] = 1 - 1 / (1 + (left / i) ** (2 * 20)) * 1 / (1 + ( i/right) ** (2 * 20))
        filter[i + N - left - right] = 1 - 1 / (1 + ( left / i) ** (2 * 20)) *  1 / (1 + ( i/right) ** (2 * 20))
        phase += 2 * cmath.pi / len(range(left, right))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    fig, (ax4, ax5, ax6, ax7) = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))

    ax1.plot(freq_axis, abs(filter))
    ax1.set_xlabel('f, c')
    ax1.set_ylabel('U, мВ')

    spectrum = ifft(filter)
    # spectrum -= spectrum.imag * 1j
    # spectrum = abs(spectrum)

    for i in range(len(spectrum)):
        if time_axis[i] > 50 and (time_axis[i] < (time_axis[len(spectrum) - 1] - 50)):
            spectrum[i] = 0

    ax2.plot(np.arange(len(spectrum)), spectrum.real)


    filter = fft(spectrum)

    ax3.plot(freq_axis, abs(filter))


    ax4.plot(np.arange(len(sig)), sig)


    fourier_of_sig = fft(sig)

    ax5.plot(freq_axis, abs(fourier_of_sig))


    fourier_of_sig = fourier_of_sig * abs(filter)

    ax6.plot(freq_axis, abs(fourier_of_sig))

    filtered_sig = ifft(fourier_of_sig)
    ax7.plot(freq_axis, filtered_sig)


    plt.show()


def HighButterWorthFilter(sig, border):
    freq_axis = fftfreq(N, 1.0 /FD)
    time_axis = np.arange(N)
    filter = np.zeros(len(freq_axis), dtype=complex)

    phase = 0

    print(freq_axis[border])
    # for i in range(2*(int(N/2) - border )):
    #     filter[i + int(N/2) - (int(N/2) - border)] = 1

    for i in range(2*(int(N/2) - border )):
        filter[len(freq_axis) - 1 - i] = 1 - 1 / (1 + (i / border) ** (2 * 20))  # cmath.rect(1, phase)
        filter[i] = 1 -  1 / (1 + (i / border) ** (2 * 20))
        phase += 2 * cmath.pi / (2*(int(N/2) - border ))

    fig, (ax1, ax2, ax3, ax8) = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
    fig, (ax4, ax5, ax6, ax7) = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))

    ax1.plot(freq_axis, abs(filter))

    spectrum = fft(filter)
    for i in range(len(spectrum)):
        if time_axis[i] > 50 and time_axis[i] < (N - 50):
            spectrum[i] = 0

    ax2.plot(spectrum.real)
    filter = ifft(spectrum)
    ax3.plot(freq_axis, filter)

    ax4.plot(np.arange(len(sig)), sig)

    fourier_of_sig = fft(sig)
    ax5.plot(freq_axis, abs(fourier_of_sig))

    fourier_of_sig = fourier_of_sig * abs(filter)

    ax6.plot(freq_axis, abs(fourier_of_sig))

    filtered_sig = ifft(fourier_of_sig)

    ax7.plot(time_axis, filtered_sig)

    ax8.plot(freq_axis, abs(fft(sig-filtered_sig)))

    plt.show()



def LowButterWorthFilter(sig, border):

    freq_axis = fftfreq(N,1. / FD)
    time_axis = np.arange(N)
    filter = np.zeros(len(freq_axis), dtype=complex)

    phase = 2 * cmath.pi / border

    print(freq_axis[border])
    for i in range(border):
        filter[ len(freq_axis)-1 - i] =  1 / (1 + ( i/border) ** (2 * 20)) # cmath.rect(1, phase)
        filter[i] =  1 / (1 + (i/border ) ** (2 * 20))
        phase += 2 * cmath.pi / border

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
    fig, (ax4, ax5, ax6, ax7) = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))

    ax1.plot(freq_axis, np.abs(filter))

    spectrum = ifft(filter)

    for i in range(len(spectrum)):
        if time_axis[i] > 100 and (time_axis[i] < (time_axis[len(spectrum) - 1] - 100)):
            spectrum[i] = 0

    ax2.plot(np.arange(len(spectrum)), spectrum.real)
    ax2.set_xlabel('t, s')
    ax2.set_ylabel('U, мВ')

    filter = fft(spectrum)

    ax3.plot(freq_axis, abs(filter))
    ax3.set_xlabel('f, c')
    ax3.set_ylabel('U, мВ')

    ax4.plot(np.arange(len(sig)), sig)
    ax4.set_xlabel('t, c')
    ax4.set_ylabel('U, мВ')

    fourier_of_sig = fft(sig)

    ax5.plot(freq_axis, abs(fourier_of_sig))

    fourier_of_sig = fourier_of_sig * abs(filter)

    ax6.plot(freq_axis, abs(fourier_of_sig))
    filtered_sig = ifft(fourier_of_sig)
    ax7.plot(np.arange(len(filtered_sig)), filtered_sig)
    plt.show()


if __name__ == '__main__':
    freq = fftfreq(N, 1. / FD)
    print(freq)
    signal = np.asarray([6. * sin(2*cmath.pi * t*100/FD) + 3. * sin(2*cmath.pi * t*500/FD) + sin(2*cmath.pi * t*1000/FD) for t in freq])
    #LowButterWorthFilter(signal, 100)

    #HighButterWorthFilter(signal, 800)
    BandButterWorthFilter(signal, 450, 550)
