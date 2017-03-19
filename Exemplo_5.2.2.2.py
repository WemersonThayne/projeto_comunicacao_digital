import numpy as np
import matplotlib.pyplot as plt

A = 1.2  # amplitude of signal
Q = 1/10  # quantization stepsize
N = 2000  # number of samples

def uniform_midrise_quantizer(x, Q):
    # limiter
    x = np.copy(x)
    idx = np.where(np.abs(x) >= 1)
    x[idx] = np.sign(x[idx])
    # linear uniform quantization
    xQ = Q * (np.floor(x/Q) + .5)

    return xQ

# generate signal
x = A * np.sin(2*np.pi/N * np.arange(N))
# quantize signal
xQ = uniform_midrise_quantizer(x, Q)
# plot signals
plot_signals(x, xQ)