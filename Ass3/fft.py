import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt


def func(x, A, freq, phase):
    return -A * np.sin(x * freq + phase)


data = np.loadtxt("oscillator.dat")

t = data[:, 0]
m = data[:, 1:4]
abs = np.sqrt(np.sum(np.square(m), axis=1))
avg = np.append(np.average(m, axis=0), np.average(abs))
print(avg)


# Interpolate
xvals = np.linspace(min(t), max(t), t.shape[0])
inter = np.interp(xvals, t, m[:, 0])

plt.plot(xvals, inter)
plt.show()


# Optional fitting

# p0 = [0.5, 2e10, 1e9]
# param, pcov = curve_fit(
#     func,
#     xvals[round(t.shape[0]) :],
#     inter[round(t.shape[0]) :],
#     p0,
#     bounds=([0.4, 1e10, 1e7], [1, 5e10, 1e10]),
# )
# print(p0)
# print(param)
# y = func(xvals[round(t.shape[0]) :], *param)
# plt.plot(xvals[round(t.shape[0]) :], y)
# plt.plot(xvals[round(t.shape[0]) :], func(xvals[round(t.shape[0]) :], *p0))
# plt.show()


# FFT
s = 8000
X = xvals[s:]
Y = inter[s:]

# Problem conditioning signalstarts close to zero and ends close to it. Sort of hard coded tho...
zer = np.where(np.abs(Y) < 0.002)[0]
id1 = int(zer[1])
id2 = int(zer[-1])


wx = X[id1:id2]
wy = Y[id1:id2]

plt.plot(wx, wy)
plt.show()

yf = np.fft.fft(wy)
freq = np.fft.fftfreq(wx.size, d=1 / (wx[1] - wx[0]))

plt.plot(freq, yf)
plt.show()

# I'm just not getting anywhere here. Sure doesn't seem right... Anyway...
