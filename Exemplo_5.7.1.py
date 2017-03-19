import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def A_law_compander(x):
    A = 87.6
    y = np.zeros_like(x)
    idx = np.where(np.abs(x) < 1/A)
    y[idx] = A*np.abs(x[idx]) / (1 + np.log(A))
    idx = np.where(np.abs(x) >= 1/A)
    y[idx] = (1 + np.log(A*np.abs(x[idx]))) / (1 + np.log(A))

    return np.sign(x)*y

def A_law_expander(y):
    A = 87.6
    x = np.zeros_like(y)
    idx = np.where(np.abs(y) < 1/(1+np.log(A)))
    x[idx] = np.abs(y[idx])*(1+np.log(A)) / A
    idx = np.where(np.abs(y) >= 1/(1+np.log(A)))
    x[idx] = np.exp(np.abs(y[idx])*(1+np.log(A))-1)/A

    return np.sign(y)*x

def uniform_midtread_quantizer(x, w):
    # quantization step
    Q = 1/(2**(w-1))
    # limiter
    x = np.copy(x)
    idx = np.where(x <= -1)
    x[idx] = -1
    idx = np.where(x > 1 - Q)
    x[idx] = 1 - Q
    # linear uniform quantization
    xQ = Q * np.floor(x/Q + 1/2)

    return xQ

def evaluate_requantization(x, xQ):
    e = xQ - x
    # SNR
    SNR = 10*np.log10(np.var(x)/np.var(e))
    print('SNR: %f dB'%SNR)
    # normalize error
    e = .2 * e / np.max(np.abs(e))
    return e


x = np.linspace(-1, 1, 2**16)
y = A_law_compander(x)
yQ4 = uniform_midtread_quantizer(y, 4)
yQ8 = uniform_midtread_quantizer(y, 8)
xQ4 = A_law_expander(yQ4)
xQ8 = A_law_expander(yQ8)

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(x, yQ4, label=r'$w=4$ bit')
plt.plot(x, yQ8, label=r'$w=8$ bit')
plt.title('Compansion and linear quantization')
plt.xlabel(r'$x$')
plt.ylabel(r'$x_Q$')
plt.legend(loc=2)
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid()

plt.subplot(122)
plt.plot(x, xQ4, label=r'$w=4$ bit')
plt.plot(x, xQ8, label=r'$w=8$ bit')
plt.title('Overall')
plt.xlabel(r'$x$')
plt.ylabel(r'$x_Q$')
plt.legend(loc=2)
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.grid()