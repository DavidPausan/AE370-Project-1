import numpy as np
from constants import MU, A
from dynamics import f

#picking a point on the orbit
r = np.array([A,0.0,0.0])
r_norm = np.linalg.norm(r)
r_hat = r / r_norm

#Mean motion of n
n = np.sqrt(MU / r_norm**3) #orbits angular rate
print("n =", n, "1/s")

#Building G and J to get the eigvals
G = MU / r_norm**3 * (3 * np.outer(r_hat,r_hat)-np.eye(3)) #Equation G in code

J = np.zeros((6,6))
J[0:3,3:6] = np.eye(3)
J[3:6,0:3] = G

lam = np.linalg.eigvals(J)
print("eigenvalues / n 9two-body):", np.round(lam / n, 4))

#Check against the real model
#point at 45 degrees lat

v = np.sqrt(MU/A)
u = np.array([A/np.sqrt(2), 0.0, A / np.sqrt(2), 0.0, v, 0.0])

J_num = np.zeros((6,6))
h = 1e-3
for j in range(6):
    du = np.zeros(6)
    du[j] = h
    f_plus = f((u + du)[None, :], 0.0)[0]
    f_minus = f((u-du)[None,:], 0.0)[0]
    J_num[:,j] = (f_plus - f_minus) / (2 * h)

lam_num = np.linalg.eigvals(J_num)
print("eigenvalues / n (with J2):" , np.round(lam_num / n, 4))

#stiffness ration and size of dt * lam

print("stiffness ratio:", np.abs(lam).max() / np.abs(lam).min())
print("|dt*lambda| at dt = 30 s:", 30 * np.abs(lam).max())




