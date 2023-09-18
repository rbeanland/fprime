#%%
import numpy as np
from scipy.integrate import quad_vec
from scipy.constants import pi, c, h, e, m_e
from scipy.optimize import curve_fit

# Throughout this code, the temperature factor is referred to as M following the convention of Bird and King,
# In plots and other writing this same quanitity is referred to as B
V = 200000.0
# lorentz factor
gamma = 1 + (e*V)/(m_e*c**2)
# electron velocity
v = c*np.sqrt(1 - 1/gamma**2)

# The factor of 1/beta is not included here,
# this is to make the results independent of accelerating voltage so it may be added later
preFactor = 1e10*2*h/(m_e*c)  # Angstroms

Z = 20

#%%
# Calculation of absorptive scattering factor

#The lobato parameterisation of the the elastic scattering factors is used, here stored in a text file
lobato_array = np.loadtxt('params_reduced.txt', delimiter = ' ')
element = lobato_array[Z-1]

def lobato(s, ab):
    g = 2*s    
    f = ab[0]*(2 + ab[5]*g**2)/(1 + ab[5]*g**2)**2 + \
        ab[1]*(2 + ab[6]*g**2)/(1 + ab[6]*g**2)**2 + \
        ab[2]*(2 + ab[7]*g**2)/(1 + ab[7]*g**2)**2 + \
        ab[3]*(2 + ab[8]*g**2)/(1 + ab[8]*g**2)**2 + \
        ab[4]*(2 + ab[9]*g**2)/(1 + ab[9]*g**2)**2
    return f

def integrand(sx, sy, s, M, ab):
    s1 = np.sqrt((s/2 + sx)**2 + sy**2)
    s2 = np.sqrt((s/2 - sx)**2 + sy**2)
    s_square = sx**2 + sy**2 - (s**2/4)
    result = lobato(s1, ab)*lobato(s2, ab)*(1 - np.exp(-2*M*s_square))
    return result

def integral1(sy, s, M, ab):
    return quad_vec(integrand, 0, np.inf, args=(sy, s, M, ab))[0]

# quad_vec is used instead of something like dblquad so that 2d arrays of s and M may be calculated efficiently
def fprime(s, M, ab):
    return preFactor*4*quad_vec(integral1, 0, np.inf, args=(s, M, ab))[0]
#%%
# loading fit functions
def gauss(x, a, b):
    return a*np.exp(-abs(b)*x**2)

# this was only used as a quick way of testing how many gaussians would be appropriate
def testcurve(x, *args):
    S = np.zeros(len(x))
    for j in range(len(args)//2):
        S += gauss(x, *args[j*2 : (j+1)*2])
    return S + args[len(args)-1]

# this is the main fitting function
def curve(x, *args):
    #print(*args)
    f = gauss(x, args[0], args[1]) + \
        gauss(x, args[2], args[3]) + \
        gauss(x, args[4], args[5]) + \
        gauss(x, args[6], args[7]) + args[8]
    return f

# This was used to print results in the case that fitting failed for an M value halfway through
# so that you may copy paste the parameters that were successfully created without having to redo anything
def printer(array):
    line = ''
    for i in range(len(array)):
        line += str(array[i]) + ','
    line = line[:-1]
    print(line)
    return None

#%%
# Loading initial coefficients
# Quite a lot of time was spent deciding which M values would be appropriate and sufficient to give a good linear interpolation
# For values of M smaller than 0.1 then a high density of different values are required to produce an acceptable degree of accuracy,
# it was assumed that, in practice, the absorptive scattering factor would be set to zero for M factors this low and it was not worth the parameterisation

# 0.01, 0.0175, 0.0275, 0.037, 0.043, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3
# 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4
Mvals = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4])

# these are both arrays of s values
svals = np.linspace(0, 4, 100)
finex = np.linspace(0, 6, 100)

Smesh, Mmesh = np.meshgrid(svals, Mvals)
zvals = fprime(Smesh, Mmesh, element)
# since negative values of z are deemed unphysical, we compromise by truncating the absorptive form factor at 0
zpos = np.where(zvals > 0, zvals, 0)


#%%
# import data from an element
# since neighbouring elements have relatively similar scattering factors,
# the previously fitted element is usually appropriate for an initial guess
importfilename = "element " + str(Z-1) + ".txt"
paramsT = np.loadtxt(importfilename, delimiter = ',')

#%%
# slice arrays
# this cell is usually skipped and only run when the fitting,
# for a certain M value failed and different initial values are required
t = 11
Mvals = Mvals[t:]
paramsT = paramsT[t:]
zvals = zvals[t:]
#%%
# fitting params

bounds = [[-np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf, 0, -np.inf],
          [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0]]


# these bounds are appropriate for heavier elements and were introduced to avoid unnecessarily thin gaussian widths
bounds2 = [[-5, 0.01, -5, 0.01, -5, 0.01, -5, 0.01, -10],
          [10, 20, 10, 20, 10, 20, 10, 20, 0]]

coefflist = []
clippedlist = []
for i in range(len(Mvals)):
    zslice = zvals[i]
    cut = np.where(zslice < 0)[0]
    if len(cut) != 0:
        cut = cut[0]
        zslice = zslice[:cut]
        #zslice = np.concatenate((zslice, np.array([0])))
    clippeds = svals[:len(zslice)]
    popt, pconv = curve_fit(curve, clippeds, zslice, p0 = paramsT[i], maxfev=5000, bounds=bounds)
    for j in range(len(popt)//2):
        popt[2*j+1] = abs(popt[2*j+1])
    #popt[len(popt)-1] = -abs(popt[len(popt)-1])
    coeffs = popt
    printer(coeffs)
    lastcurve = i
    coefflist.append(coeffs)
    clippedlist.append(clippeds)

coeff_array = np.array(coefflist)
filename = "element " + str(Z) + ".txt"
np.savetxt(filename, coeff_array, fmt='%s', delimiter = ',')

#%%
# load last fitted params into paramsT
# this is run if the fit failed for a certain M value,
# in which case the parameters for the previous M value of the current element are tried
prevbest = coefflist[lastcurve]
paramsT[lastcurve:] = prevbest