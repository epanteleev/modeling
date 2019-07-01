from scipy import *
from scipy.signal import *
import matplotlib.pyplot as plt
import numpy as np
import math


def _buttap(N):
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    p = -np.exp(1j * pi * m / (2 * N))
    return z, p


def _iirfilter(N, Wn, fs=None):
    z, p = _buttap(N)
    k = 1
    warped = 2 * fs * np.tan(pi * Wn / fs)
    z, p, k = lp2lp_zpk(z, p, k, wo=warped)
    z, p, k = bilinear_zpk(z, p, k, fs=fs)
    return zpk2sos(z, p, k)


def butter(N, Wn, fs=None):
    return _iirfilter(N, Wn, fs=fs)

FD = 2000
N = 1000



sig = [math.sin(2 * math.pi * 100* i / FD ) + math.sin(2 * math.pi * 700 * i / FD) for i in range(N)]

#sig = [100 < i < 200 for i in range(N)]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
freq = np.fft.rfftfreq(N, 1. / FD)

ax1.plot(np.arange(N) / float(FD), sig)

t = [i for i in range(N)]
sos = butter(10, 130, fs=FD)
filtered = sosfilt(sos, sig)

ax2.plot(t, filtered)
ax2.set_title('15 Hz high-pass')
ax2.set_xlabel('Time [seconds]')

spec = np.fft.rfft(sig)

ax3.plot(freq, np.abs(spec)/N)
ax3.set_title('Amplitude')
#ax3.set_xlabel('Frequencies')

res_spec = np.fft.rfft(filtered)
ax4.plot(freq, np.abs(res_spec)/N)
ax4.set_title('Amplitude')
#ax4.set_xlabel('Frequencies')
plt.tight_layout()
plt.show()

b, a = butter(4, 100, fs=FD)
w, h = freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()