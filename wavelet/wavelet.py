from math import *
import matplotlib.pyplot as plt
N = 600


res = [sin(2 *pi*k/200) for k in range(N)]

X = [i for i in range(N)]

for k in range(N):
    print (res[k])

def discreteHaarWaveletTransform(x):
    N = len(x)
    output = [0.0]*N

    length = N//2
    while True:
        for i in range(0,length):
            output[i] = x[i * 2] + x[i * 2 + 1]
            output[length + i] = x[i * 2] - x[i * 2 + 1]

        if length == 1:
            return output

        x = output[:length << 1]

        length = length//2


sig_res = discreteHaarWaveletTransform(res)

for k in range(N):
    print (res[k])

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

ax1.plot(X, res, 'b')
ax1.set_title('Spectrum result')

ax1.set_xlabel('Freq, Hz')
ax1.set_ylabel('Amplitude')

ax2.plot(X, sig_res, 'b')
ax2.set_title('Spectrum result')

ax2.set_xlabel('Freq, Hz')
ax2.set_ylabel('Amplitude')
plt.show()