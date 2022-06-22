import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import linalg
from scipy.linalg import inv, solve, eigh

L = 1
nb_elem = 5
nb_vec = np.arange(nb_elem)
nb_noeud = nb_elem + 1
elem_length = L / nb_elem
node_i, node_j = np.arange(0, nb_noeud - 1, 1), np.arange(1, nb_noeud, 1)
xi, yi = np.arange(0, L, elem_length), np.arange(0, L, elem_length) * 0
xj, yj = np.arange(elem_length, L + elem_length, elem_length), np.arange(elem_length, L + elem_length, elem_length) * 0

INC = np.vstack((nb_vec,
                 node_i,
                 node_j,
                 xi,
                 yi,
                 xj,
                 yj))

cx = np.vstack((xi, xj))
cy = np.vstack((yi, yj))
plt.figure()
plt.plot(cx, cy, '-o')
plt.xlabel("length")
plt.axis('equal')
plt.show()

# MATERIAL
E = 2.1E11
rho = 7800
b = 0.01
h = 0.03
A = h * b
Iz = h ** 3 * b / 12

#
n = nb_elem
ngl = 2
GL = (n + 1) * ngl
Kg = np.zeros((GL, GL))
Mg = np.zeros((GL, GL))
Ug = np.zeros((GL, 1))
In = np.eye(GL)
Fg = np.zeros((GL, 1))
Fy = 1
Fg[len(Fg) - 1] = Fy


def Matrice(n, INC, E, A, Iz, rho, Kg, Mg, In):
    for i in range(0, n):
        le = np.sqrt((INC[5, i] - INC[3, i]) ** 2 + (INC[6, i] - INC[4, i]) ** 2)
        c, s = int((INC[5, i] - INC[3, i]) / le), int((INC[6, i] - INC[4, i]) / le)
        Rot = np.array([[c, s, 0, 0],
                        [-s, c, 0, 0],
                        [0, 0, c, s],
                        [0, 0, -s, c]])
        k = E * Iz / le ** 3 * np.array([[12, 6 * le, -12, 6 * le],
                                         [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                         [-12, -6 * le, 12, -6 * le],
                                         [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        m = (rho * A * le) / 420 * np.array([[156, 22 * le, 54, -13 * le],
                                             [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                             [54, 13 * le, 156, -22 * le],
                                             [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
        k_re = Rot.T @ k @ Rot
        m_re = Rot.T @ m @ Rot
        a = In[2 * i:2 * i + 4, :]
        Kg = Kg + a.T @ k_re @ a
        Mg = Mg + a.T @ m_re @ a
    return Kg, Mg

K, M = Matrice(n, INC, E, A, Iz, rho, Kg, Mg, In)
C = np.array([0,1])
for i  in [0,1]:
    M = np.delete(M, C, axis= i)
    K = np.delete(K, C, axis=i)
F = np.delete(Fg, C, 0)

plt.figure()
plt.spy(K, marker = None, markersize=4)
plt.show()

freq = np.linspace(0, 100, 100, endpoint=True)
wf = freq*2*np.pi
U = np.zeros((len(wf),len(F)))

for i in range(len(wf)):
    U[i,:] = np.linalg.solve(K - wf[i]**2*M,F).T

plt.figure()
plt.plot(freq, 20*np.log10(np.abs(U[:,::2])))
plt.xlabel("$Frequency$")
plt.ylabel("$Reponse (dB)$")
plt.show()

wn2, phi = eigh(K,M)

wn = np.sqrt(wn2)
print("frequence naturelle (Hz) : ")
wnHz = wn/(2*np.pi)
print(wnHz[0:6])

Meff = phi.T @ M @ phi
phi_n = phi @ np.diag(1.0/np.sqrt(np.diag(Meff)))

nb_mode = 5
coor_x = np.append(xi, xj[-1])
Phisc = np.vstack((np.zeros((len(C), int(GL-2))),phi_n))

plt.figure()
plt.plot(coor_x,Phisc[:,0:nb_mode],'-o')
plt.xlabel('lenght (m)')
plt.ylabel('ampl')
plt.show()
