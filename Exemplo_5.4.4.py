import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

idx = 130000  # index to start plotting

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
    # plot signals
    plt.figure(figsize=(10, 4))
    plt.plot(x[idx:idx+100], label=r'signal $x[k]$')
    plt.plot(xQ[idx:idx+100], label=r'requantized signal $x_Q[k]$')
    plt.plot(e[idx:idx+100], label=r'quantization error $e[k]$')
    plt.xlabel(r'$k$')
    plt.legend()
    # normalize error
    e = .2 * e / np.max(np.abs(e))
    return e

# load speech sample
x, fs = sf.read('speech.wav')
x = x/np.max(np.abs(x))


xQ = uniform_midtread_quantizer(x, 2)
e = evaluate_requantization(x, xQ)
sf.write('speech_2bit.wav', xQ, fs)
sf.write('speech_2bit_error.wav', e, fs)