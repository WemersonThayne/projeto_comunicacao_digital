import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

w = 16  # wordlength of the quantized signal
L = 2**np.arange(1,10)  # oversampling factors

N = 8192  # length of signals
Om0 = 100*2*np.pi/N  # frequency of harmonic signal
Q = 1/(2**(w-1))  # quantization step


def uniform_midtread_quantizer(x, Q):
    # limiter
    x = np.copy(x)
    idx = np.where(x <= -1)
    x[idx] = -1
    idx = np.where(x > 1 - Q)
    x[idx] = 1 - Q
    # linear uniform quantization
    xQ = Q * np.floor(x/Q + 1/2)

    return xQ

def SNR_oversampled_ADC(L):
    x = (1-Q)*np.cos(Om0*np.arange(N))
    xu = (1-Q)*np.cos(Om0*np.arange(N*L)/L)
    # quantize signal
    xQu = uniform_midtread_quantizer(xu, Q)
    # low-pass filtering and decimation
    xQ = sig.resample(xQu, N)
    # estimate SNR
    e = xQ - x

    return 10*np.log10((np.var(x)/np.var(e)))


# compute SNR for oversampled ADC
SNR = [SNR_oversampled_ADC(l) for l in L]

# plot result
plt.figure(figsize=(10, 4))
plt.semilogx(L, SNR, label='with oversampling')
plt.plot(L, (6.02*w+1.76)*np.ones(L.shape), label='without oversampling' )
plt.xlabel(r'oversampling factor $L$')
plt.ylabel(r'SNR in dB')
plt.legend(loc='upper left')
plt.grid()