###############################################################################
# //////////////////////////////////////////////////////////////////////////////
# head--------------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###############################################################################

#__::((odmkGaussian1.py))::__

# Python basic Gaussian signal generator

###############################################################################
# //////////////////////////////////////////////////////////////////////////////
# main--------------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###############################################################################



# -*- coding: utf-8 -*-

# from matplotlib import pyplot as mp
import matplotlib.pyplot as plt
import numpy as np


# Create dictionary of math text examples:
mathtext_gaussian = {
    0: r"$g(x) = A*e^{\left ( -\frac{(x-\mu )^{2}}{2*\sigma ^{2}} \right )} +d$",

    1: r"$g(x) = A*exp \left ( -\frac{(x-\mu )^{2}}{2*\sigma ^{2}} \right ) +d$",

    2: r"$\frac{3}{4},\ \binom{3}{4},\ \stackrel{3}{4},\ "
    r"\left(\frac{5 - \frac{1}{x}}{4}\right),\ \ldots$",

    3: r"$\sqrt{2},\ \sqrt[3]{x},\ \ldots$",

    4: r"$\mathrm{Roman}\ , \ \mathit{Italic}\ , \ \mathtt{Typewriter} \ "
    r"\mathrm{or}\ \mathcal{CALLIGRAPHY}$",

    5: r"$\acute a,\ \bar a,\ \breve a,\ \dot a,\ \ddot a, \ \grave a, \ "
    r"\hat a,\ \tilde a,\ \vec a,\ \widehat{xyz},\ \widetilde{xyz},\ "
    r"\ldots$",

    6: r"$\alpha,\ \beta,\ \chi,\ \delta,\ \lambda,\ \mu,\ "
    r"\Delta,\ \Gamma,\ \Omega,\ \Phi,\ \Pi,\ \Upsilon,\ \nabla,\ "
    r"\aleph,\ \beth,\ \daleth,\ \gimel,\ \ldots$",

    7: r"$\coprod,\ \int,\ \oint,\ \prod,\ \sum,\ "
    r"\log,\ \sin,\ \approx,\ \oplus,\ \star,\ \varpropto,\ "
    r"\infty,\ \partial,\ \Re,\ \leftrightsquigarrow, \ \ldots$"}

# define colors
# http://www.rapidtables.com/web/color/RGB_Color.htm
mplot_black = (0./255., 0./255., 0./255.)
mplot_white = (255./255., 255./255., 255./255.)
mplot_red = (255./255., 0./255., 0./255.)
mplot_orange = (255/255., 165/255., 0./255.)
mplot_darkorange = (255/255., 140/255., 0./255.)
mplot_orangered = (255/255., 69/255., 0./255.)
mplot_yellow = (255./255., 255./255., 0./255.)
mplot_lime = (0./255., 255./255., 0./255.)
mplot_green = (0./255., 128./255., 0./255.)
mplot_darkgreen = (0./255., 100./255., 0./255.)
mplot_cyan = (0./255., 255./255., 255./255.)
mplot_blue = (0./255., 0./255., 255./255.)
mplot_midnightblue = (25./255., 25./255., 112./255.)
mplot_magenta = (255./255., 0./255., 255./255.)
mplot_grey = (128./255., 128./255., 128./255.)
mplot_silver = (192./255., 192./255., 192./255.)
mplot_purple = (128./255., 0./255., 128./255.)
mplot_maroon = (128./255., 0./255., 0./255.)
mplot_olive = (128./255., 128./255., 0./255.)
mplot_teal = (0./255., 128./255., 128./255.)


# //////////////////////////////////////////////////////////////////////////////
# setup & definitions end------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# //////////////////////////////////////////////////////////////////////////////
# function definitions begin---------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# 1D Gaussian function
# A = amplitude (Gain)
# mu = position at the center of the peak
# sig = (standard deviation) controls the width of the Bell (Gaussian RMS width)
# d = the value that the function asymptotically approaches far from the peak
#     (d most often set to zero)

def gaussian(x, A, mu, sig, d):
    return A*np.exp(-np.power(x - mu, 2.) / (2*np.power(sig, 2.))) + d

# rhs = np.exp(-.5 * (x**2 + y**2) / sigma**2)

# //////////////////////////////////////////////////////////////////////////////
# function definitions end-----------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# //////////////////////////////////////////////////////////////////////////////
# Gaussian Ex1 begin-----------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
# Gaussian functions with amplitude of 5 centered at 500
    
A = 5
d = 0
# d = -0.00001

xLin = np.linspace(0, 40, 1000)

# plt.figure(1)
fig = plt.figure(num=1, facecolor='olive', edgecolor='k')

# implement using zip and logarithmic spacing
# write gaussian data into matrix array before -> plot matrix
for mu, sig in [(20, 0.5), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5)]:
    plt.plot(gaussian(xLin, A, mu, sig, d))

plt.xlabel('x position')
plt.ylabel('Amplitude')
plt.title('ODMK Gaussian Example 3')
gaussian1D = mathtext_gaussian[1]
plt.annotate(gaussian1D,
             xy=(500, 5.3),
             xycoords='data', color=mplot_orangered, ha='center',
             fontsize=20)
             
plt.axis([0, 1000, 0, 6])
# plt.grid(True)
plt.grid(color='c', linestyle=':', linewidth=.5)
ax = plt.gca()
ax.set_facecolor("k")             
             
# //////////////////////////////////////////////////////////////////////////////
# gaussian Ex1 end--------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# plt.figure(3)
# fig = plt.figure(num=3, facecolor='olive', edgecolor='k')
# odmkLinePlt = plt.plot(t1, sig1, t2, sig2)
# odmkLinePlt1 = plt.plot(t1, sig1)
# odmkLinePlt2 = plt.plot(t2, sig2)
# plt.setp(odmkLinePlt1, color='m', ls='-', marker='o', mfc='r', linewidth=3.00)
# plt.setp(odmkLinePlt2, color='g', ls=':', marker='^', mfc='b', linewidth=3.00)
#
# plt.text(1.3, .54, r'$\aleph\ = f(t): red dots$', color='black', fontsize=16)
# plt.text(1.3, .44, r'$\beth\ = -f(t): blue triangles$', color='black', fontsize=16)


# //////////////////////////////////////////////////////////////////////////////
# Gaussian Ex2 begin-----------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def gaussKernel(sideLen=5, gSigma=1.0):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    axisX = np.linspace(-(sideLen - 1) / 2.0, (sideLen - 1) / 2.0, sideLen)
    xx, yy = np.meshgrid(axisX, axisX)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(gSigma))

    return kernel / np.sum(kernel)


gaussK = gaussKernel(5, 1)


# plt.figure(2)
fig = plt.figure(num=2, facecolor='olive', edgecolor='k')
plt.imshow(gaussK, interpolation='none')

# //////////////////////////////////////////////////////////////////////////////
# Gaussian Ex4 begin-----------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

U = 60
m = np.linspace(-0.5, 0.5, U)    # 60 points between -1 and 1
delta = m[1] - m[0]              # delta^2 is the area of each grid cell
(x, y) = np.meshgrid(m, m)       # Create the mesh

sigma = 0.1
norm_constant = 1 / (2 * np.pi * sigma**2)

rhs = np.exp(-.5 * (x**2 + y**2) / sigma**2)
ker = norm_constant * rhs
print(ker.sum() * delta**2)


# plt.figure(4)
fig = plt.figure(num=4, facecolor='olive', edgecolor='k')

plt.contour(x, y, ker)
# plt.axis('equal')
plt.axis([-0.3, 0.3, -0.3, 0.3])

# temp print
for xi in x[0]:
    print(xi)




# //////////////////////////////////////////////////////////////////////////////
# Gaussian Ex5 begin-----------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# wrapped bi-variate gaussian distribution (??? questionsable... ???)

if 0:

    U = 100
    Ukern = np.copy(U)
    # Ukern = 15

    m = np.arange(U)
    i = m.reshape(U, 1)
    j = m.reshape(1, U)

    sigma = 2.0
    ii = np.minimum(i, Ukern-i)
    jj = np.minimum(j, Ukern-j)
    xmu = (ii-0) / sigma
    ymu = (jj-0) / sigma
    ker = np.exp(-.5 * (xmu**2 + ymu**2))
    ker /= np.abs(ker).sum()

    ''' Point Density '''
    ido = np.random.randint(U, size=(10, 2)).astype(np.int)
    og = np.zeros((U, U))
    np.add.at(og, (ido[:, 0], ido[:, 1]), 1)

    ''' Convolution via FFT and inverse-FFT '''
    v1 = np.fft.fft2(ker)
    v2 = np.fft.fft2(og)
    v0 = np.fft.ifft2(v2*v1)
    dd = np.abs(v0)


    # plt.figure(5)
    fig = plt.figure(num=5, facecolor='olive', edgecolor='k')

    plt.plot(ido[:, 1], ido[:, 0], 'ko', alpha=.3)
    plt.imshow(dd, origin='upper')


plt.show()
