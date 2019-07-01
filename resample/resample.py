import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
from math import *
from numpy.fft import rfft, rfftfreq, irfft

freq = 3000
freq_new = 25

N = 8000


newN = freq_new * int(N/freq)


def LCM(a, b):
    m = a * b
    while a != 0 and b != 0:
        if a > b:
            a %= b
        else:
            b %= a
    return m // (a + b)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpassfilter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def resample(signal, N_old, N_new):
    lcm_size = LCM(N_old, N_new)
    print(lcm_size)
    extend_signal = np.zeros(lcm_size)

    for i in range(N_old):
        extend_signal[i * (lcm_size // N_old)] = signal[i]

    if N_old < N_new:
        filtered = lowpassfilter(extend_signal, freq, lcm_size)
    else:
        filtered = lowpassfilter(extend_signal, freq_new, lcm_size)

    new_signal = np.zeros(N_new)
    for i in range(len(extend_signal)):
        times = (lcm_size // N_new)
        new_signal[i // times] = filtered[i]
    return new_signal


if __name__ == '__main__':

    extend_size = LCM(N, newN)

    div_times = extend_size / newN

    sig = [sin(2 * pi * i * 100 / N) + sin(2 * pi* i * 50 / N) + sin(2 * pi * i * 160 / N) for i in range(N)]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(16, 10))

    ax1.plot(range(len(sig)), sig)

    new_signal = resample(sig, N, newN)

    ax2.plot(range(len(new_signal)),new_signal)

    e = rfft(new_signal)
    ax3.plot(rfftfreq(newN, 1.0 / newN ), abs(e))

    ax4.plot(rfftfreq(N, 1.0 / N)[:len(e)], abs(rfft(sig))[:len(e)])
    plt.show()
