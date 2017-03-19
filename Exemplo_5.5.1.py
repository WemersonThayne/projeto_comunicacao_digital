import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

w = 8  # wordlength of the quantized signal
A = 1  # amplitude of input signal
N = 32768  # number of samples


def uniform_midtread_quantizer_w_ns(x, Q):
    # limiter
    x = np.copy(x)
    idx = np.where(x <= -1)
    x[idx] = -1
    idx = np.where(x > 1 - Q)
    x[idx] = 1 - Q
    # linear uniform quantization with noise shaping
    xQ = Q * np.floor(x/Q + 1/2)
    e = xQ - x
    xQ = xQ - np.concatenate(([0], e[0:-1]))

    return xQ[1:]


# quantization step
Q = 1/(2**(w-1))
# compute input signal
x = np.random.uniform(size=N, low=-A, high=(A-Q))
# quantize signal
xQ = uniform_midtread_quantizer_w_ns(x, Q)
e = xQ - x[1:]
# estimate PSD of error signal
nf, Pee = sig.welch(e, nperseg=64)
# estimate SNR
SNR = 10*np.log10((np.var(x)/np.var(e)))
print('SNR = %f in dB' %SNR)


plt.figure(figsize=(10,5))
Om = nf*2*np.pi
plt.plot(Om, Pee*6/Q**2, label='simulated')
plt.plot(Om, np.abs(1 - np.exp(-1j*Om))**2, label='theory')
plt.plot(Om, np.ones(Om.shape), label='w/o noise shaping')
plt.title('Estimated PSD of quantization error')
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$\hat{\Phi}_{e_H e_H}(e^{j \Omega}) / \sigma_e^2$')
plt.axis([0, np.pi, 0, 4.5]);
plt.legend(loc='upper left')
plt.grid()