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


# load speech sample
x, fs = sf.read('speech_8k.wav')
x = x/np.max(np.abs(x))

# linear quantization
xQ = uniform_midtread_quantizer(x, 8)
e = evaluate_requantization(x, xQ)
sf.write('speech_8k_8bit.wav', xQ, fs)
sf.write('speech_8k_8bit_error.wav', e, fs)

# A-law quantization
y = A_law_compander(x)
yQ = uniform_midtread_quantizer(y, 8)
xQ = A_law_expander(yQ)
e = evaluate_requantization(x, xQ)
sf.write('speech_Alaw_8k_8bit.wav', xQ, fs)
sf.write('speech_Alaw_8k_8bit_error.wav', e, fs)