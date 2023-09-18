#%%
#This code was used to check the accuracy of the parameters after they had been fitted
# loading functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from scipy.constants import pi, c, h, e, m_e
from scipy.interpolate import RegularGridInterpolator

# Accelerating voltage, volts
V = 100000.0
# lorentz factor
gamma = 1 + (e*V)/(m_e*c**2)
# electron velocity
v = c*np.sqrt(1 - 1/gamma**2)
# wavelength
wav = 1e10* h/(gamma*m_e*v) # Angstroms

preFactor = 1e10*2*h/(m_e*c)  # Angstroms

Z = 28



lobato_array = np.loadtxt('c:\\Users\\u2106849\\Documents\\URSS\\code\\params_reduced.txt', delimiter = ' ')
carbon = lobato_array[5]
gallium = lobato_array[30]
gold = lobato_array[78]
lithium = lobato_array[2]

element = lobato_array[Z - 1]

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

def fprime(s, M, ab):
    return preFactor*4*quad_vec(integral1, 0, np.inf, args=(s, M, ab))[0]
#%%
# loading fit functions
def gauss(x, a, b):
    return a*np.exp(-b*x**2)

def curve(x, *args):
    S = np.zeros(len(x))
    for i in range(len(args)//2):
        S += gauss(x, *args[i*2 : (i+1)*2])
    return S + args[len(args)-1]
#%%
# 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4
Mvals = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4])
svals = np.linspace(0, 4, 50)
finex = np.linspace(0, 6, 100)
filename = "element " + str(Z) + ".txt"
coefflist = np.loadtxt(filename, delimiter = ',')



#%%
#first the curves were plotted to check if there was any anomalous behaviour in the parameterisation
# plotting gaussians
colours = plt.cm.jet(np.linspace(0, 1, 6))
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("fit comparisons", fontsize=15)
ax.set_xlabel("s", fontsize=20)
ax.set_ylabel("f'", fontsize=20)

svals = np.linspace(0, 6, 50)
Smesh, Mmesh = np.meshgrid(svals, Mvals)
zvals = fprime(Smesh, Mmesh, element)
zpos = np.where(zvals > 0, zvals, 0)


fitlist = []
for i in range(len(Mvals)):
    fitdata0 = curve(finex, *coefflist[i])
    fitdata0 = np.where(fitdata0 > 0, fitdata0, 0)
    ax.plot(svals, zpos[i], label= 'actual function', color = colours[1])
    ax.plot(finex, fitdata0, label= 'fit function', color = colours[4])
    fitlist.append(fitdata0)
fit_array = np.asarray(fitlist)


#%%
# interpolation between curves
finex = np.linspace(0, 6, 100)
finey = np.linspace(0, 4, 100)
fitdata0 = np.vstack((np.zeros(100), fit_array))
Mvals0 = np.concatenate((np.array([0]), Mvals))
interp = RegularGridInterpolator((finex, Mvals0), np.transpose(fitdata0))

fineX, fineY = np.meshgrid(finex, finey)
interpZ = interp((fineX, fineY))
interpZ_pos = np.where(interpZ > 0 , interpZ, 0)
#%%
# comparison data
compareZ = fprime(fineX, fineY, element)
compareZpos = np.where(compareZ > 0, compareZ, np.nan)

#%%
# error plot
# a heat map was also created to check the accuracy of the linear interpolation between curves
error = abs(compareZpos - interpZ_pos)/compareZpos
fzeros = np.where(compareZ <= 0, 0, np.nan)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("4 gaussians and a constant" + " parameterisation element " + str(Z), fontsize=15)
ax.set_xlabel("s (Å⁻¹)", fontsize=15)
ax.set_ylabel("B (Å²)", fontsize=15)
plot = ax.pcolormesh(fineX, fineY, error, cmap='plasma', vmin=0.01, vmax=0.05)
ax.pcolormesh(fineX, fineY, fzeros, cmap='inferno')
fig.colorbar(plot)

#%%
