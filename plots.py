#%%
from time import process_time, perf_counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from scipy.constants import pi, c, h, e, m_e
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

Z = 31
lobato_array = np.loadtxt('params_reduced.txt', delimiter = ' ')
element = lobato_array[Z-1]
print(element)

preFactor = 1e10*2*h/(m_e*c)

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


s = 0
M = 0.7

r = np.linspace(0, 2, 200)
p = np.linspace(70*pi/180 + 0.698, 70*pi/180 - 0.698 + 2*pi, 200)
R, P = np.meshgrid(r, p)
X, Y = R*np.cos(P), R*np.sin(P)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= (20,20))

#ax.set_proj_type('ortho')
z = integrand(X, Y, s, M, element)
ax.plot_surface(X, Y, z, alpha=1)

#ax.plot_wireframe(X, Y, z, color = 'k', linewidth=0.5)
ax.view_init(elev=20, azim=70)
ax.set_xlim([-2.2, 2.2])
ax.set_ylim([-2.2, 2.2])
ax.set_zlim([0, 0.7])
ax.tick_params(axis='x', which='major', labelsize=15, pad=10)
ax.tick_params(axis='y', which='major', labelsize=15, pad=10)
ax.tick_params(axis='z', which='major', labelsize=15, pad=10)
ax.dist = 9

#%%
svals = np.linspace(0, 4, 100)
Bvals = np.linspace(0, 2, 100)
X, Y = np.meshgrid(svals, Bvals)
f = fprime(X, Y, element)


#%%
xticks = np.arange(0, 4.5, 0.5)
yticks = np.arange(0, 2.25, 0.25)
zticks = np.arange(-350, 50, 50)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= (10, 10), layout="constrained")
ax.dist = 13
ax.plot_wireframe(X, Y, f, alpha=1, linewidth=0.5)
ax.set_xlabel("s (Å⁻¹)", fontsize=12, labelpad=12)
ax.set_ylabel("B (Å²)", fontsize=12, labelpad=12)
ax.set_zlabel("f' (Å)", fontsize=12, labelpad=12)
ax.set_zscale("log")
ax.tick_params(axis='z', which='major', pad=5)

# %%
f = np.where(f>0, f, 0)

xticks = np.arange(0, 4.5, 0.5)
yticks = np.arange(0, 2.25, 0.25)
zticks = np.arange(0, 0.018, 0.002)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= (10, 10), layout="tight")
ax.dist = 13
ax.plot_wireframe(X, Y, f, alpha=1, linewidth=0.5)
ax.set_xlabel("s (Å⁻¹)", fontsize=12, labelpad=12)
ax.set_ylabel("B (Å²)", fontsize=12, labelpad=12)
ax.set_zlabel("f' (Å)", fontsize=12, labelpad=18)
ax.tick_params(axis='z', which='major', pad=8)

# %%
#0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4
Mvals = [0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 2.75, 4]


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= (10, 10), layout="tight")

ax.view_init(azim=-40)
ax.dist = 12
for M in Mvals:
    z = fprime(svals, M, element)
    z = np.where(z>=0, z, np.nan)
    ax.plot(svals, np.full(100, M), z, "r")

x = np.linspace(0, 4, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
f = fprime(X, Y, element)
f = np.where(f>0, f, 0)

ax.plot_wireframe(X, Y, f, alpha=0.9, linewidth=0.5)
ax.set_xlabel("s (Å⁻¹)", fontsize=12, labelpad=8)
ax.set_ylabel("B (Å²)", fontsize=12, labelpad=8)
ax.set_zlabel("f' (Å)", fontsize=12, labelpad=18)
ax.tick_params(axis='z', which='major', pad=10)
# %%
Bvals = np.array([0.01, 0.05, 0.1, 0.15, 0.2])
svals = np.linspace(0, 6, 100)
fig, ax = plt.subplots(figsize= (10, 10), layout="tight")
ax.set_xlabel("s (Å⁻¹)", fontsize=12)
ax.set_ylabel("f' (Å)", fontsize=12)
for B in Bvals:
    ydata = fprime(svals, B, Z)
    ax.plot(svals, ydata, label="B =" + str(B))
plt.legend()

# %%
