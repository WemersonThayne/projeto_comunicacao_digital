import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

w = 8  # wordlength of the quantized signal
Pc = np.logspace(-20, np.log10(.5), num=500)  # probabilities for clipping
N = int(1e6)  # number of samples


def compute_SNR(Pc):
    # compute input signal
    sigma_x = - np.sqrt(2) / np.log(Pc)
    x = np.random.laplace(size=N, scale=sigma_x/np.sqrt(2) )
    # quantize signal
    xQ = uniform_midtread_quantizer(x, Q)
    e = xQ - x
    # compute SNR
    SNR = 10*np.log10((np.var(x)/np.var(e)))

    return SNR


# quantization step
Q = 1/(2**(w-1))
# compute SNR for given probabilities
SNR = [compute_SNR(P) for P in Pc]

# plot results
plt.figure(figsize=(8,4))
plt.semilogx(Pc, SNR)
plt.xlabel('Probability for clipping')
plt.ylabel('SNR in dB')
plt.grid()