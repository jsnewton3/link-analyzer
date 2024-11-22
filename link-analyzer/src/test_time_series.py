import random
from math import sin, exp
import matplotlib as plot
import matplotlib.pyplot as plt
import numpy as np
from operator import add, sub

x = np.linspace(0, 15, 150)


def sin_on_linspace(x, period, amp, omega, is_stationary=False):
    y = []
    ops = (add, sub)
    for ind, val in enumerate(x):
        if not is_stationary:
            if random.random() < .1:
                op = random.choice(ops)
                amp = op(amp, amp * .2)
                omega = op(omega, omega * .2)
        y.append(amp * sin(omega * val + period))
    return np.array(y)


def exp_on_linspace(x, amp, sigma):
    y = []
    for ind, val in enumerate(x):
        y.append(amp * exp(val * sigma))
    return np.array(y)


y = ((sin_on_linspace(x, period=2, amp=50, omega=8) +
      sin_on_linspace(x, period=1, amp=40, omega=2)) +
     exp_on_linspace(x, amp=1, sigma=.4))
plt.plot(x, y)
plt.show()
np.save("../test_ts.npy", [x, y])
