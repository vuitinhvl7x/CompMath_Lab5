import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import optimize
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from numpy import double
import warnings
warnings.filterwarnings("ignore",
                        message="Support for FigureCanvases without a required_interactive_framework attribute was deprecated in Matplotlib 3.6 and will be removed two minor releases later.")


print("WELCOME TO ADAM'S METHOD OF SOLVING DIFFERENTIAL EQUATIONS")
print("CHOOSE A FUNCTION BELOW BY THE INDICES")
print("1. y' = x + y -1 \n"
      "2. y' = e^x - tan y\n"
      "3. y' = x^3 -3")

choose_function = int(input("Enter your choice here: "))
interval = np.array(list(map(float, input('Enter the interval [x0, xn]: ').strip().split()))[:2])
if interval[0] == interval[1]:
    print('intervals can not be equal')
    exit(0)
if interval[0] > interval[1]:
    swap = interval[0]
    interval[0] = interval[1]
    interval[1] = swap

y0 = float(input("Enter the value of y(x0): "))
accuracy = abs(int(input('Enter order of accuracy. The least is 4 ')))
if accuracy < 4:
    accuracy = 4
print("----------------USING ADAM'S METHOD----------------")
# Implementing Runge-Kutta's method into Adam's method as the single step method
# setting accuracy O(h(4))
h = (interval[1] - interval[0]) / accuracy
Y = []
Y.append(y0)
dY = []
x = interval[0]
X = []
for i in range(0, accuracy + 1):
    X.append(x)
    x += h

if choose_function == 1:
    for i in range(0, accuracy - 1):
        k1 = X[i] + Y[i] - 1
        k2 = ((X[i] + h / 2) + (Y[i] + h * k1 / 2)) - 1
        k3 = ((X[i] + h / 2) + (Y[i] + h * k2 / 2)) - 1
        k4 = ((X[i] + h) + (Y[i] + h * k3)) - 1
        Y.append(Y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)


    for i in range(0, accuracy):
        dY.append(X[i] + Y[i] - 1)
        # x += h
    Y.append(Y[len(Y) - 1] + h * (
            55 * dY[len(dY) - 1] - 59 * dY[len(dY) - 2] + 37 * dY[len(dY) - 3] - 9 * dY[len(dY) - 4]) / 24)
    dY.append(interval[1] + Y[len(Y) - 1] - 1)

if choose_function == 2:
    for i in range(0, accuracy -1):
        k1 = math.exp(x) - math.tan(Y[i])
        k2 = math.exp(x + h / 2) - math.tan(Y[i] + k1 / 2)
        k3 = math.exp(x + h / 2) - math.tan(Y[i] + k2 / 2)
        k4 = math.exp(x + h) - math.tan(Y[i] + k3)
        Y.append(Y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    for i in range(0, accuracy):
        dY.append(math.exp(x) - math.tan(Y[i]))
        x += h
        print(dY)
    Y.append(Y[len(Y) - 1] + h * (
            55 * dY[len(dY) - 1] - 59 * dY[len(dY) - 2] + 37 * dY[len(dY) - 3] - 9 * dY[len(dY) - 4]) / 24)
    dY.append(math.exp(interval[1]) + math.tan(Y[len(Y) - 1]))

if choose_function == 3:
    for i in range(0, accuracy - 1):
        k1 = X[i] + Y[i] - 1
        k2 = ((X[i] + h / 2) + (Y[i] + h * k1 / 2)) - 1
        k3 = ((X[i] + h / 2) + (Y[i] + h * k2 / 2)) - 1
        k4 = ((X[i] + h) + (Y[i] + h * k3)) - 1
        Y.append(Y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    for i in range(0, accuracy):
        dY.append(X[i]**3-3)
    Y.append(Y[len(Y) - 1] + h * (
            55 * dY[len(dY) - 1] - 59 * dY[len(dY) - 2] + 37 * dY[len(dY) - 3] - 9 * dY[len(dY) - 4]) / 24)
    dY.append(interval[1]**3 -3)

# Correction of y4
Y[len(Y) - 1] = Y[len(Y) - 2] + h * (
        9 * dY[len(dY) - 1] + 19 * dY[len(dY) - 2] - 5 * dY[len(dY) - 3] + dY[len(dY) - 4]) / 24

print('X values = ')
print(X)
print('Y values = ')
print(Y)
print('dY values = ')
print(dY)


# Plot the graph
xplt = np.linspace(X[0], X[-1])
dxplt = xplt
yplt = np.array([], double)
dyplt = np.array([], double)
xp = 1
dxp =1
for xp in xplt:
    yp = 0
    for i in range(len(Y)):
        p=1
        for j in range(len(X)):
            if j != i:
                p*=(xp - X[j])/(X[i] - X[j])
        yp += Y[i]*p
    yplt = np.append(yplt, yp)
for dxp in dxplt:
    dyp = 0
    for i in range(len(dY)):
        p=1
        for j in range(len(X)):
            if j != i:
                p*=(dxp - X[j])/(X[i] - X[j])
        dyp += dY[i]*p
    dyplt = np.append(dyplt, dyp)


plt.plot(X, Y, 'ro')
plt.plot(X, dY, 'bo')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid()
label1 = 'y = (x^2/2) +x*(y-1) + ' + str(y0)
label2 = 'y = e^x - x*tan(y) + ' + str(y0)
label3 = 'y = x^4/4 -3x + ' + str(y0)
if choose_function == 1:
    plt.plot(xplt, yplt, color='blue', label=label1)
    plt.plot(xplt, dyplt, linestyle='dashed', color='green', label="y' = x+y-1")
if choose_function == 2:
    plt.plot(xplt, yplt, color='blue', label=label2)
    plt.plot(xplt, dyplt, linestyle='dashed', color='green', label="y' = e^x - tan(y)")
if choose_function == 3:
    plt.plot(xplt, yplt, color='blue', label=label3)
    plt.plot(xplt, dyplt, linestyle='dashed', color='green', label="y' = x^3 -3")
plt.legend()
plt.show()


