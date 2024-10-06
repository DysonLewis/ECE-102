import numpy
import matplotlib.pyplot as plt

# (a) Plotting e^(-t) * cos(2πt) for -10 ≤ t ≤ 10 with step size 0.2
t_a = numpy.arange(-10, 10.2, 0.2)
y_a = numpy.exp(-t_a) * numpy.cos(2 * numpy.pi * t_a)

# (b) Implementing ReLU function for -5 ≤ t ≤ 5 with step size 0.1
def relu(t):
    return numpy.maximum(0, t)

t_b = numpy.arange(-5, 5.1, 0.1)
y_b = relu(t_b)

# Function to compute the square of a function f
def square(t, f):
    return f(t) * f(t)

# Function to compute the even part of a function f
def even(t, f):
    return (f(t) + f(-t)) / 2

# Function to compute the odd part of a function f
def odd(t, f):
    return (f(t) - f(-t)) / 2

# Time vector for plotting square, even, and odd parts of ReLU
t = numpy.arange(-5, 5.1, 0.1)

# Compute values
y_square = [square(tt, relu) for tt in t]
y_even = [even(tt, relu) for tt in t]
y_odd = [odd(tt, relu) for tt in t]

# Create a figure with 4 subplots (2x2 layout)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First subplot: e^(-t) * cos(2πt)
axs[0, 0].plot(t_a, y_a, label=r'$e^{-t} \cos(2\pi t)$')
axs[0, 0].set_title(r'Plot of $e^{-t} \cos(2\pi t)$ for $-10 \leq t \leq 10$')
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('y')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Second subplot: ReLU function
axs[0, 1].plot(t_b, y_b, label=r'ReLU function', color='orange')
axs[0, 1].set_title('Plot of ReLU function for $-5 \leq t \leq 5$')
axs[0, 1].set_xlabel('t')
axs[0, 1].set_ylabel('ReLU(t)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Third subplot: Square of ReLU
axs[1, 0].plot(t, y_square, label="Square of ReLU", color='green')
axs[1, 0].set_title("Square of ReLU")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("Square of ReLU(t)")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Fourth subplot: Even and Odd parts of ReLU
axs[1, 1].plot(t, y_even, label="Even part of ReLU", color='blue')
axs[1, 1].plot(t, y_odd, label="Odd part of ReLU", color='red')
axs[1, 1].set_title("Even and Odd parts of ReLU")
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("Value")
axs[1, 1].grid(True)
axs[1, 1].legend()

# Adjust layout for better display
plt.tight_layout()
plt.show()
