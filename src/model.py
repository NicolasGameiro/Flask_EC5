# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:41:46 2022

@author: ngameiro
"""

"""[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""

# Mesh
from prettytable import PrettyTable as pt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = (8, 6)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from matplotlib.patches import Rectangle, Polygon

# Model
from mesh import Mesh
from matplotlib.animation import FuncAnimation
from numpy.linalg import inv
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from tracer_maison_3d import *

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import time, datetime


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class FEM_Model():
    def __init__(self, mesh, E=210E9):
        self.mesh = mesh
        self.E = E
        if self.mesh.dim == 2:
            self.mesh.node_list, self.mesh.element_list, self.mesh.name, self.mesh.color, self.mesh.Section = self.mesh.maillage()
            self.load = np.zeros([len(self.mesh.node_list), 3])
            self.bc = np.eye(len(self.mesh.node_list) * 3)
            self.U = np.zeros(len(self.mesh.node_list) * 3)
            self.React = np.zeros(len(self.mesh.node_list) * 3)
            self.S = np.empty((0, 4))
        elif self.mesh.dim == 3:
            self.load = np.zeros([len(self.mesh.node_list), 6])
            self.bc = np.eye(len(self.mesh.node_list) * 6)
            self.U = np.zeros(len(self.mesh.node_list) * 6)
            self.React = np.zeros(len(self.mesh.node_list) * 6)
            self.S = np.empty((0, 7))
        self.dist_load = np.array([[1, 2, 0]])
        self.lbc = []

    def test(self):
        self.mesh.geom()

    def apply_load(self, node_load, node):
        if node > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")
        elif (len(node_load) == 3) or (len(node_load) == 6):
            self.load[node - 1, :] = node_load
            # print("nodal load applied")
            if self.mesh.debug == True:
                print(self.load)
        else:
            print("Error : uncorrect load format")

    def apply_distributed_load(self, q, element):
        L = self.get_length(element)
        if self.mesh.dim == 2:
            Q = np.array([0,
                          -q * L / 2,
                          -q * L ** 2 / 12,
                          0,
                          -q * L / 2,
                          q * L ** 2 / 12])
            self.load[element[0] - 1] = self.load[element[0] - 1] + Q[:3]
            self.load[element[1] - 1] = self.load[element[1] - 1] + Q[3:6]
        elif self.mesh.dim == 3:
            Q = np.array([0, 0, -q * L / 2, -q * L ** 2 / 12, 0, 0,
                          0, 0, -q * L / 2, q * L ** 2 / 12, 0, 0])
            self.load[element[0] - 1] = self.load[element[0] - 1] + Q[:6]
            self.load[element[1] - 1] = self.load[element[1] - 1] + Q[6:12]
        self.dist_load = np.append(self.dist_load, [[element[0], element[1], q]], axis=0)
        # print(self.dist_load)

    def apply_bc(self, node_bc, node):
        if node > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")
        elif len(node_bc) == 3:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 3 * (node - 1))
            # print("boundary condition applied")
        elif len(node_bc) == 6:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 6 * (node - 1))
            # print("boundary condition applied")
        else:
            print("Error : uncorrect bc format")

    def Rot(self, c, s):
        """ Rotation matrix in 2D
        """
        Rotation_matrix = np.array([[c, -s, 0, 0, 0, 0],
                                    [s, c, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, c, -s, 0],
                                    [0, 0, 0, s, c, 0],
                                    [0, 0, 0, 0, 0, 1]])
        return Rotation_matrix

    def Rot_3D(self, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        RR = np.identity(12)
        vec1 = [1, 0, 0]
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        RR[0:3, 0:3] = rotation_matrix
        RR[3:6, 3:6] = rotation_matrix
        RR[6:9, 6:9] = rotation_matrix
        RR[9:12, 9:12] = rotation_matrix
        return RR

    def mini_rot(self, c, s):
        R = np.array([[c, s],
                      [-s, c]])
        return R

    def K_elem(self, L_e, h, b):
        S = h * b * 1e-4
        I = b * h ** 3 / 12 * 1e-8
        K_elem = self.E / L_e * np.array([[S, 0, 0, -S, 0, 0],
                                          [0, 12 * I / L_e ** 2, 6 * I / L_e, 0, -12 * I / L_e ** 2, 6 * I / L_e],
                                          [0, 6 * I / L_e, 4 * I, 0, -6 * I / L_e, 2 * I],
                                          [-S, 0, 0, S, 0, 0],
                                          [0, -12 * I / L_e ** 2, -6 * I / L_e, 0, 12 * I / L_e ** 2, -6 * I / L_e],
                                          [0, 6 * I / L_e, 2 * I, 0, -6 * I / L_e, 4 * I]])
        return K_elem

    def stress_2(self):
        S = self.mesh.S
        I = self.mesh.Iy
        h = 0.22
        self.sig = np.zeros([len(self.mesh.node_list), 3])
        for i in range(len(self.mesh.node_list)):
            # en MPa
            self.sig[i, 0] = self.load[i, 0] / S / 1e6  # traction/compression (en MPa)
            self.sig[i, 1] = self.load[i, 1] / S / 1e6  # cisaillement (en MPa)
            self.sig[i, 2] = self.load[i, 2] / I * (h / 2) / 1e6  # flexion (en MPa)
        print(self.sig)

    def K_elem_3d(self, L: float, h: float, b: float, E: float = 1, nu: float = 0.3, ay: float = 0,
                  az: float = 0) -> np.array:
        """ Calcul de la matrice de raideur avec prise en compte de l'énergie cisaillement avec les termes ay et az.

        :param L: longueur de l'element
        :type L: float
        :param E: Module d'Young
        :type E: float
        :param G: Module de coulomb
        :type G: float
        :param J: Module de torsion
        :type J: float
        :param ay:
        :type ay:
        :param az:
        :type az:
        :return: matrice de raideur en 3D
        :rtype: np.array
        """
        G = 1  # E/2/(1+nu)
        S = 1  # h * b
        Iy = 1  # b * h ** 3 / 12
        Iz = 1  # h * b ** 3 / 12
        J = 1  # Iy + Iz
        Ktc = E * S / L
        KT = G * J / L
        Kf1 = 12 * E * Iz / (L ** 3 * (1 + az))
        Kf2 = 12 * E * Iy / (L ** 3 * (1 + ay))
        Kf3 = -6 * E * Iy / (L ** 2 * (1 + ay))
        Kf4 = 6 * E * Iz / (L ** 2 * (1 + az))
        Kf5 = (4 + ay) * E * Iy / (L * (1 + ay))
        Kf6 = (4 + az) * E * Iz / (L * (1 + az))
        Kf7 = (2 - ay) * E * Iy / (L * (1 + ay))
        Kf8 = (2 - az) * E * Iz / (L * (1 + az))
        K_elem = np.array([[Ktc, 0, 0, 0, 0, 0, -Ktc, 0, 0, 0, 0, 0],  # 1
                           [0, Kf1, 0, 0, 0, Kf4, 0, -Kf1, 0, 0, 0, Kf4],
                           [0, 0, Kf2, 0, Kf3, 0, 0, 0, -Kf2, 0, Kf3, 0],
                           [0, 0, 0, KT, 0, 0, 0, 0, 0, -KT, 0, 0],
                           [0, 0, Kf3, 0, Kf5, 0, 0, 0, -Kf3, 0, Kf7, 0],
                           [0, Kf4, 0, 0, 0, Kf6, 0, -Kf4, 0, 0, 0, Kf8],
                           [-Ktc, 0, 0, 0, 0, 0, Ktc, 0, 0, 0, 0, 0],  # 7
                           [0, -Kf1, 0, 0, 0, -Kf4, 0, Kf1, 0, 0, 0, -Kf4],
                           [0, 0, -Kf2, 0, -Kf3, 0, 0, 0, Kf2, 0, -Kf3, 0],
                           [0, 0, 0, -KT, 0, 0, 0, 0, 0, KT, 0, 0],
                           [0, 0, Kf3, 0, Kf7, 0, 0, 0, -Kf3, 0, Kf5, 0],
                           [0, Kf4, 0, 0, 0, Kf8, 0, -Kf4, 0, 0, 0, Kf6]], dtype='float')
        return K_elem

    def changement_base(self, P, M):
        return P.dot(M).dot(np.transpose(P))

    def changement_coord(self):
        BB = []
        for i in range(len(self.mesh.element_list)):  # Une matrice de changement de coord par element
            # print("generation de la matrice de passage de l'element ", i + 1, ":")
            B = np.zeros([len(self.mesh.node_list) * 3, 6])
            noeud1 = self.mesh.element_list[i, 0]
            noeud2 = self.mesh.element_list[i, 1]
            B[(noeud1 - 1) * 3, 0] = 1
            B[(noeud1 - 1) * 3 + 1, 1] = 1
            B[(noeud1 - 1) * 3 + 2, 2] = 1
            B[(noeud2 - 1) * 3, 3] = 1
            B[(noeud2 - 1) * 3 + 1, 4] = 1
            B[(noeud2 - 1) * 3 + 2, 5] = 1
            BB.append(B)
        return BB

    def changement_coord_3D(self):
        BB = []
        for i in range(len(self.mesh.element_list)):  # Une matrice de changement de coord par element
            # print("generation de la matrice de passage de l'element ", i + 1, ":")
            B = np.zeros([len(self.mesh.node_list) * 6, 12])
            noeud1 = self.mesh.element_list[i, 0]
            noeud2 = self.mesh.element_list[i, 1]
            B[(noeud1 - 1) * 6, 0] = 1
            B[(noeud1 - 1) * 6 + 1, 1] = 1
            B[(noeud1 - 1) * 6 + 2, 2] = 1
            B[(noeud1 - 1) * 6 + 3, 3] = 1
            B[(noeud1 - 1) * 6 + 4, 4] = 1
            B[(noeud1 - 1) * 6 + 5, 5] = 1
            ###
            B[(noeud2 - 1) * 6, 0] = 1
            B[(noeud2 - 1) * 6 + 1, 1] = 1
            B[(noeud2 - 1) * 6 + 2, 2] = 1
            B[(noeud2 - 1) * 6 + 3, 3] = 1
            B[(noeud2 - 1) * 6 + 4, 4] = 1
            B[(noeud2 - 1) * 6 + 5, 5] = 1
            BB.append(B)
        return BB

    def get_length(self, element):
        noeud1 = element[0]
        noeud2 = element[1]
        if self.mesh.dim == 2:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        elif self.mesh.dim == 3:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            z_1 = self.mesh.node_list[noeud1 - 1, 2]
            z_2 = self.mesh.node_list[noeud2 - 1, 2]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2 + (z_2 - z_1) ** 2)
        return L_e

    def get_angle(self, element):
        """ Return the cosinus and the sinus associated with the angle of the element
        in the global coordinate

        :return: tuple with cosinus and sinus
        :rtype: 2-uple
        """
        noeud1 = element[0]
        noeud2 = element[1]
        if self.mesh.dim == 2:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
            c = (x_2 - x_1) / L_e
            s = (y_2 - y_1) / L_e
        elif self.mesh.dim == 3:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            z_1 = self.mesh.node_list[noeud1 - 1, 2]
            z_2 = self.mesh.node_list[noeud2 - 1, 2]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2 + (z_2 - z_1) ** 2)
            c = (x_2 - x_1) / L_e
            s = (y_2 - y_1) / L_e
        return c, s

    def get_bc(self):
        """Return the boundary condition in a matrix format

        :return: matrix with 1 if the dof is blocked and 0 if the dof is free
        :rtype: np.array
        """
        BC = np.zeros(3 * len(self.mesh.node_list))
        for i in self.lbc:
            BC[i] = 1
        BC = BC.reshape((len(self.mesh.node_list), 3))
        return BC

    def assemblage_2D(self):
        """ Return the global stiffness matrix of the mesh

        :return: matrix of size(dll*3*nb_node,dll*3*nb_node)
        :rtype: np.array

        """
        BB = self.changement_coord()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL) * 3, len(NL) * 3])
        for i in range(len(EL)):
            element = EL[i]
            L_e = self.get_length(element)
            c, s = self.get_angle(element)
            rot = self.Rot(c, s)
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            # rotation matrice elem
            K_rot = rot.dot(self.K_elem(L_e, h, b)).dot(np.transpose(rot))
            M_global = M_global + self.changement_base(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.K_elem(L_e, h, b))
                print(K_rot)
        return M_global

    def assemblage_3D(self):
        """ Return the global stiffness matrix of the mesh

        :return: matrix of size(dll*3*nb_node,dll*3*nb_node)
        :rtype: np.array

        """
        BB = self.changement_coord_3D()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL) * 6, len(NL) * 6])
        for i in range(len(EL)):
            element = EL[i]
            L_e = self.get_length(element)
            print(element)
            rot = self.Rot_3D(NL[element[1] - 1])
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            print(rot, self.K_elem_3d(L_e, h, b))

            # rotation matrice elem
            K_rot = rot.dot(self.K_elem_3d(L_e, h, b)).dot(np.transpose(rot))
            M_global = M_global + self.changement_base(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.K_elem_3d(L_e, h, b))
                print(K_rot)
        return M_global

    def solver_frame(self):
        self.bc = np.delete(self.bc, self.lbc, axis=1)
        if self.mesh.dim == 2:
            K_glob = self.assemblage_2D()
        elif self.mesh.dim == 3:
            K_glob = self.assemblage_3D()
        K_glob_r = np.transpose(self.bc).dot(K_glob).dot(self.bc)
        ### en cas de matrice singuliaire
        m = 0
        K_glob_r = K_glob_r + np.eye(K_glob_r.shape[1]) * m
        ###
        F = np.vstack(self.load.flatten())
        F_r = np.transpose(self.bc).dot(F)
        U_r = inv(K_glob_r).dot(F_r)
        self.U = self.bc.dot(U_r)
        self.React = K_glob.dot(self.U) - F

    def get_stress(self, elem):  # bien prendre les valeurs dans le repère local de l'element
        NL = self.mesh.node_list
        node_i, node_j = elem[0] - 1, elem[1] - 1
        L = self.get_length(elem)
        U = self.U
        G = 81E9
        h, b = 0.1, 0.1  # self.mesh.Section[i,0], self.mesh.Section[i,1]
        Iy = b * h ** 3 / 12
        Iz = h * b ** 3 / 12
        k = 5 / 6
        if self.mesh.dim == 2:
            epsilon_x = (U[3 * node_j] - U[3 * node_i]) / L
            sigma_x = self.E * epsilon_x / 1E6
            sigma_fy = self.E * h * (U[3 * node_j + 2] - U[3 * node_i + 2]) / L / 1E6
            tau_y = np.array([0]) / 1E6
            sigma_VM = np.sqrt((sigma_x + sigma_fy) ** 2 + 3 * (tau_y) ** 2) / 1E6
            sigma_T = np.sqrt((sigma_x + sigma_fy) ** 2 + 4 * (tau_y) ** 2) / 1E6
            if self.mesh.debug == True:
                print("déformation (en mm) =", epsilon_x[0] * 1E3)
                print("contrainte normale (en MPa) =", sigma_x[0])
                print("contrainte normale de flexion (en MPa) =", sigma_fy[0])
                print("contrainte cisaillement de flexion (en MPa) =", tau_y[0])
                print("contrainte Von Mises (en MPa) =", sigma_VM[0])
                print("contrainte Tresca (en MPa) =", sigma_T[0])
            return np.array([sigma_x, sigma_fy, tau_y, sigma_VM, sigma_T])
        elif self.mesh.dim == 3:
            RR = self.Rot_3D(NL[elem[1] - 1])
            rot_max = RR[0:6, 0:6]
            Ui = np.transpose(rot_max).dot(U[6 * node_j: 6 * node_j + 6])
            Uj = np.transpose(rot_max).dot(U[6 * node_j: 6 * node_j + 6])
            epsilon_x = (Uj[0] - Ui[0]) / L
            sigma_x = self.E * epsilon_x
            tau_x = G * (Uj[3] - Ui[3]) / L * max(h, b)
            sigma_fy = self.E * h * (U[6 * node_j + 5] - U[6 * node_i + 5]) / L
            sigma_fz = self.E * b * (U[6 * node_j + 4] - U[6 * node_i + 4]) / L
            Ay = 12 * self.E * Iy / (k * G * h * b * L ** 2 + 12 * self.E * Iy)
            Az = 12 * self.E * Iz / (k * G * h * b * L ** 2 + 12 * self.E * Iz)
            tau_y = -G * Ay * (2 * U[6 * node_i + 1] + U[6 * node_i + 5] * L - 2 * U[6 * node_j + 1] + U[
                6 * node_j + 5] * L) / L ** 2
            tau_z = -G * Az * (2 * U[6 * node_i + 2] + U[6 * node_i + 4] * L - 2 * U[6 * node_j + 2] + U[
                6 * node_j + 4] * L) / L ** 2
            sigma_VM = np.sqrt((sigma_x + sigma_fy + sigma_fz) ** 2 + 3 * (tau_x + tau_y + tau_z) ** 2)
            if self.mesh.debug == True:
                print("contrainte normale (en MPa) =", sigma_x[0] / 1E6)
                print("contrainte normale de flexion (en MPa) =", sigma_fz[0] / 1E6)
                print("contrainte normale de flexion (en MPa) =", sigma_fy[0] / 1E6)
                print("contrainte cisaillement de torsion (en MPa) =", tau_x[0] / 1E6)
                print("contrainte cisaillement de flexion (en MPa) =", tau_y[0] / 1E6)
                print("contrainte cisaillement de flexion (en MPa) =", tau_z[0] / 1E6)
                print("contrainte Von Mises (en MPa) =", sigma_VM[0] / 1E6)
            return np.array([sigma_x, sigma_fy, sigma_fz, tau_x, tau_y, tau_z, sigma_VM])

    def stress(self):
        EL = self.mesh.element_list
        for elem in EL:
            self.S = np.append(self.S, self.get_stress(elem))
        return self.S

    def get_res(self):
        self.res = {}
        self.res['U'] = []
        self.res['React'] = []
        self.res['node'] = []
        self.res['elem'] = []
        if self.mesh.dim == 2:
            for i in range(len(self.mesh.node_list)):
                self.res['U'].append(
                    {'node': i + 1, 'Ux': self.U[i * 3], 'Uy': self.U[3 * i + 1], 'phi': self.U[3 * i + 2]})
                self.res['React'].append(
                    {'node': i + 1, 'Fx': self.React[i][0], 'Fy': self.React[i + 1][0], 'Mz': self.React[i + 2][0]})
                self.res['node'].append({'node': i + 1, 'X': self.mesh.node_list[i][0], 'Y': self.mesh.node_list[i][1]})
                # self.res['elem'].append({'elem' : i + 1 , 'node i' : self.mesh.element_list[i][0] , 'node j' : self.mesh.element_list[i][1]})
        elif self.mesh.dim == 3:
            for i in range(len(self.mesh.node_list)):
                self.res['U'].append(
                    {'node': i + 1, 'Ux': self.U[6 * i], 'Uy': self.U[6 * i + 1], 'Uz': self.U[6 * i + 2],
                     'thx': self.U[6 * i + 3], 'thy': self.U[6 * i + 4], 'thz': self.U[6 * i + 5]})
                self.res['React'].append(
                    {'node': i + 1, 'Fx': self.React[i][0], 'Fy': self.React[i + 1][0], 'Mz': self.React[i + 2][0]})
                self.res['node'].append({'node': i + 1, 'X': self.mesh.node_list[i][0], 'Y': self.mesh.node_list[i][1]})
                # self.res['elem'].append({'elem' : i + 1 , 'node i' : self.mesh.element_list[i][0] , 'node j' : self.mesh.element_list[i][1]})
        return self.U, self.React, self.res

    def charge_2D(self, pt1, pt2, q):
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        dx, dy = x2 - x1, y2 - y1
        L = np.sqrt(dx ** 2 + dy ** 2)
        a = np.arctan(dy / dx)

        nb_pt = 5
        amplitude = 1
        x = np.linspace(pt1[0], pt2[0], nb_pt)
        y = np.linspace(pt1[1], pt2[1], nb_pt)

        ax = plt.subplot(111)
        for i in range(0, nb_pt):
            plt.arrow(x[i],  # x1
                      y[i] + amplitude,  # y1
                      0,  # x2 - x1
                      -amplitude,  # y2 - y1
                      color='r',
                      lw=1,
                      length_includes_head=True,
                      head_width=0.02,
                      head_length=0.05,
                      zorder=6)
        plt.plot([pt1[0], pt2[0]], [pt1[1] + 1, pt2[1] + 1], lw=1, color='r', zorder=6)
        ax.text(x1 + dx / 2 * 0.9, y1 + dy / 2 + amplitude * 1.2,
                "q = " + str(q / 1000) + " kN/m",
                size=10, zorder=2, color="k")
        x = [pt1[0], pt2[0], pt2[0], pt1[0], pt1[0]]
        y = [pt1[1], pt2[1], pt2[1] + 1, pt1[1] + 1, pt1[1]]

        ax.add_patch(Polygon(xy=list(zip(x, y)), fill=True, color='red', alpha=0.1, lw=0))
        return

    def charge_3D(self, ax, pt1, pt2, q):
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        z1, z2 = pt1[2], pt2[2]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # a = np.arctan(dy / dx)
        nb_pt = 10
        amplitude = 1
        x = np.linspace(x1, x2, nb_pt)
        y = np.linspace(y1, y2, nb_pt)
        z = np.linspace(z1, z2, nb_pt)

        for i in range(0, nb_pt):
            a = Arrow3D([x[i], x[i]],
                        [y[i], y[i]],
                        [z[i] + amplitude, z[i]],
                        mutation_scale=10,
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)
        line, = ax.plot([x1, x2], [y1, y2], [z1 + amplitude, z2 + amplitude], color='r', lw=1)
        ax.text(x1 + dx / 2, y1 + dy / 2, z1 + dz / 2,
                "q = " + str(q) + " kN/m",
                size=20, zorder=2, color="k")
        return

    def plot_forces(self, type='nodal', pic=False, path="./"):
        plt.figure()
        F = self.load
        NL = self.mesh.node_list
        scale_force = np.max(np.abs(F))
        x = [x for x in self.mesh.node_list[:, 0]]
        y = [y for y in self.mesh.node_list[:, 1]]
        size = 50
        offset = size / 40000.
        plt.scatter(x, y, c='y', s=size, zorder=5)
        for i, location in enumerate(zip(x, y)):
            plt.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)
        for i in range(len(self.mesh.element_list)):
            xi, xj = self.mesh.node_list[self.mesh.element_list[i, 0] - 1, 0], self.mesh.node_list[
                self.mesh.element_list[i, 1] - 1, 0]
            yi, yj = self.mesh.node_list[self.mesh.element_list[i, 0] - 1, 1], self.mesh.node_list[
                self.mesh.element_list[i, 1] - 1, 1]
            plt.plot([xi, xj], [yi, yj], color=self.mesh.color[i], lw=1, linestyle='--')
        ### Trace les efforts
        if type == 'nodal':
            plt.quiver(NL[:, 0] - F[:, 0] / scale_force, NL[:, 1] - F[:, 1] / scale_force, F[:, 0], F[:, 1], color='r',
                       angles='xy', scale_units='xy', scale=scale_force)
        elif type == 'dist':
            for elem in self.dist_load[1:]:
                pt1 = self.mesh.node_list[elem[0] - 1]
                pt2 = self.mesh.node_list[elem[1] - 1]
                self.charge_2D(pt1, pt2, elem[2])
        plt.grid()
        plt.ylim([-1, max(x)])
        plt.xlim([-1, max(y)])
        plt.axis('equal')
        # plt.show()
        if pic:
            plt.savefig(path + 'load.png', format='png', dpi=200)
        return

    def plot_forces3D(self, type='dist', pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # plt.gca(projection='3d')
        F = self.load
        NL = self.mesh.node_list
        scale_force = np.max(np.abs(F))
        x = [x for x in self.mesh.node_list[:, 0]]
        y = [y for y in self.mesh.node_list[:, 1]]
        z = [z for z in self.mesh.node_list[:, 2]]
        ax.scatter(x, y, z, c='y', s=100, zorder=1)
        for i, location in enumerate(zip(x, y)):
            ax.text(x[i], y[i], z[i], str(i + 1), size=20, zorder=2, color="k")
        for i in range(len(self.mesh.element_list)):
            xi, xj = self.mesh.node_list[self.mesh.element_list[i, 0] - 1, 0], self.mesh.node_list[
                self.mesh.element_list[i, 1] - 1, 0]
            yi, yj = self.mesh.node_list[self.mesh.element_list[i, 0] - 1, 1], self.mesh.node_list[
                self.mesh.element_list[i, 1] - 1, 1]
            zi, zj = self.mesh.node_list[self.mesh.element_list[i, 0] - 1, 2], self.mesh.node_list[
                self.mesh.element_list[i, 1] - 1, 2]
            line, = ax.plot([xi, xj], [yi, yj], [zi, zj], color=self.mesh.color[i], lw=1, linestyle='--')
            line.set_label(self.mesh.name[i])
        ### Trace les efforts
        if type == 'nodal':
            f_length = np.sqrt(F[:, 0] ** 2 + F[:, 1] ** 2 + F[:, 2] ** 2) / scale_force
            plt.quiver(NL[:, 0] - F[:, 0] / scale_force, NL[:, 1] - F[:, 1] / scale_force,
                       NL[:, 2] - F[:, 2] / scale_force,
                       F[:, 0] / scale_force, F[:, 1] / scale_force, F[:, 2] / scale_force, color='r', pivot="tail",
                       length=max(f_length), normalize=True)
        elif type == 'dist':
            for elem in self.dist_load[1:]:
                pt1 = self.mesh.node_list[elem[0] - 1]
                pt2 = self.mesh.node_list[elem[1] - 1]
                self.charge_3D(ax, pt1, pt2, elem[2])
        # self.charge_3D(ax, [0, 0, 2.5], [0, 6 / 2, 5], 1)
        ax.set_title("Structure")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.view_init(elev=20., azim=-20.)
        """
        x, y , z = 0, 0, 2.5+1
        u, v, w = 0, 0, -1
        ax.quiver(x, y, z, u, v, w, length=1, normalize=True)
        """
        ax.set_xlim(-1, max(self.mesh.node_list[:, 0]) + 1)
        ax.set_ylim(-1, max(self.mesh.node_list[:, 1]) + 1)
        ax.set_zlim(0, max(self.mesh.node_list[:, 2]) + 1)
        plt.tight_layout()
        plt.grid()
        if pic:
            plt.savefig(path + 'load.png', format='png', dpi=200)
        return ax

    def interpol(self, x1, x2, y1, y2, y3, y4, r):
        x3 = x1
        x4 = x2
        V = np.array([[1, x1, x1 ** 2, x1 ** 3],
                      [1, x2, x2 ** 2, x2 ** 3],
                      [0, 1, 2 * x3, 3 * x3 ** 2],
                      [0, 1, 2 * x4, 3 * x4 ** 2]])
        # print(V)
        R = np.array([y1, y2, y3, y4])
        R = np.vstack(R)
        P = np.hstack(inv(V).dot(R))
        P = P[::-1]
        p = np.poly1d([x for x in P])
        x = np.linspace(x1, x2, r)
        y = p(x)
        return x, y

    def plot_disp_f_ex(self, scale=1e4, r=150):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            plt.plot([xi, xj], [yi, yj], color='k', lw=1, linestyle='--')
        for i in range(len(EL)):
            x1 = NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale
            x2 = NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale
            y1 = NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale
            y2 = NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale
            y3 = U[(EL[i, 0] - 1) * 3 + 2]
            y4 = U[(EL[i, 1] - 1) * 3 + 2]
            L_e = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            c = np.round((x2 - x1) / L_e, 2)
            # print("c =", c)
            a = np.arccos(c) % 1
            # print("a = ", a)
            x, y = self.interpol(x1[0], x2[0], y1[0], y2[0], y3[0] + a, -y4[0] + a, r)
            x_scatter.append(x)
            y_scatter.append(y)
            color.append(np.linspace(U[(EL[i, 0] - 1) * 3 + 1], U[(EL[i, 1] - 1) * 3 + 1], r))
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        plt.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        plt.colorbar(label='disp'
                     , orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        return

    def plot_disp_f(self, scale=1e2, r=150, dir='x', pic=False, path="./"):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            plt.plot([xi, xj], [yi, yj], color='k', lw=1, linestyle='--')
        for i in range(len(EL)):
            if dir == 'y':
                plt.title("y")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0], r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3 + 1], U[(EL[i, 1] - 1) * 3 + 1], r))
            elif dir == "x":
                plt.title("x")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3], U[(EL[i, 1] - 1) * 3], r))
            elif dir == "sum":
                plt.title("sum")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3] + U[(EL[i, 0] - 1) * 3 + 1],
                                         U[(EL[i, 1] - 1) * 3] + U[(EL[i, 1] - 1) * 3 + 1], r))
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        plt.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        plt.colorbar(label='disp'
                     , orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        if pic:
            plt.savefig(path + 'res_' + dir + '.png', format='png', dpi=200)
        return

    def plot_stress(self, scale=1e4, r=100, s='sx', pic=False, path="./"):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        S = self.S
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        # maillage non deforme
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            plt.plot([xi, xj], [yi, yj], color='k', lw=1, linestyle='--')
        for i in range(len(EL)):
            n1, n2 = EL[i, 0] - 1, EL[i, 1] - 1
            if s == 'sx':
                plt.title("tensile stress (sx)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5])
            elif s == "sf":
                plt.title("bending stress (sf)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 1])
            elif s == "ty":
                plt.title("shear stress (ty)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 2])
            elif s == "s_vm":
                plt.title("Von Mises Stress (svm)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 3])
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        plt.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        plt.colorbar(label='stress', orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        if pic:
            plt.savefig(path + 'stress_' + s + '.png', format='png', dpi=200)
        return

    def plot_axis(self, elem):
        print(elem)
        NL = self.mesh.node_list
        node_i = NL[elem[0] - 1]
        node_j = NL[elem[1] - 1]
        dx, dy, dz = node_j[0] - node_i[0], node_j[1] - node_i[1], node_j[2] - node_i[2]
        vx = [dx, dy, dz]  # vecteur directeur de l'element
        RR = self.Rot_3D(vx)
        rr = RR[0:3, 0:3]
        print(rr)
        if True in np.isnan(rr):
            vy = [0, 1, 0]
            vz = [0, 0, 1]
        else:
            vy = rr * [0, 1, 0]
            vz = rr * [0, 0, 1]

        plt.quiver(node_i[0], node_i[1], node_i[2], vx[0], vx[1], vx[2], color='r', length=0.1, normalize=True)
        plt.quiver(node_i[0], node_i[1], node_i[2], vy[0], vy[1], vy[2], color='g', length=0.1, normalize=True)
        plt.quiver(node_i[0], node_i[1], node_i[2], vz[0], vz[1], vz[2], color='b', length=0.1, normalize=True)

    def plot_disp_f_3D(self, scale=1e0, r=80, dir='x', pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        z_scatter = []
        color = []
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            zi, zj = NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2]
            line, = ax.plot([xi, xj], [yi, yj], [zi, zj], color=self.mesh.color[i], lw=1, linestyle='--')
            line.set_label(self.mesh.name[i])
            self.plot_axis(EL[i, :])
        for i in range(len(EL)):
            if dir == 'y':
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0], r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 6 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 6 + 1] * scale, r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6 + 1], U[(EL[i, 1] - 1) * 6 + 1], r))
            elif dir == "x":
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 6] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6], U[(EL[i, 1] - 1) * 6], r))
            elif dir == "z":
                x_scatter.append(
                    np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2] + U[(EL[i, 0] - 1) * 6 + 2] * scale,
                                             NL[EL[i, 1] - 1, 2] + U[(EL[i, 1] - 1) * 6 + 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6 + 2], U[(EL[i, 1] - 1) * 6 + 2], r))
            elif dir == "sum":
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 6] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 6 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 6 + 1] * scale, r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2] + U[(EL[i, 0] - 1) * 6 + 2] * scale,
                                             NL[EL[i, 1] - 1, 2] + U[(EL[i, 1] - 1) * 6 + 2] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6] + U[(EL[i, 0] - 1) * 6 + 1] + U[(EL[i, 0] - 1) * 6 + 2],
                                         U[(EL[i, 1] - 1) * 6] + U[(EL[i, 1] - 1) * 6 + 1] + U[(EL[i, 1] - 1) * 6 + 2],
                                         r))
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        scat = ax.scatter3D(x_scatter, y_scatter, z_scatter, c=color, cmap=cmap, s=40, edgecolor='none')
        # ax.colorbar(label='disp', orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.colorbar(scat)
        ax.set_title("Déplacement " + dir)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_box_aspect([1, 1, 1])
        self.set_equal_aspect_3D(ax)
        plt.tight_layout()
        plt.grid()
        if pic:
            plt.savefig(path + 'res_' + dir + '.png', format='png', dpi=200)
        return

    def set_equal_aspect_3D(self, ax):
        """
        Set aspect ratio of plot correctly
        Args:
            :ax: (obj) axis object
        """

        # See https://stackoverflow.com/a/19248731
        # ax.set_aspect('equal') --> raises a NotImplementedError
        # See https://github.com/matplotlib/matplotlib/issues/1077/

        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    def __str__(self):
        return "fem solver"

    def U_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Phi (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 3 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 3 + 2], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Uz (m)", "Phix (rad)", "Phiy (rad)", "Phiz (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 6], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 4], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 5], precision=2, exp_digits=2)])
        print(tab)

    def R_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Node", "Fx (N)", "Fy (N)", "Mz (N.m)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.React[i][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i + 1][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i + 2][0], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.React[i * 6][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 1][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 2][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 3][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 4][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 5][0], precision=2, exp_digits=2)])
        print(tab)

    def S_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Elem", "Sx (MPa)", "Sf (MPa)", "Ty (MPa)", "SVM (MPa)", "Tresca (MPa)"]
            for i in range(len(self.mesh.element_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.S[i * 5], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i * 5 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i * 5 + 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i * 5 + 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i * 5 + 4], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Uz (m)", "Phix (rad)", "Phiy (rad)", "Phiz (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 6], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 4], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 5], precision=2, exp_digits=2)])
        print(tab)

    def rapport(self):
        doc = DocxTemplate("cctr_template.docx")

        im_load = InlineImage(doc, image_descriptor='load.png', width=Mm(150), height=Mm(100))
        im_res_x = InlineImage(doc, image_descriptor='res_x.png', width=Mm(150), height=Mm(100))
        im_res_y = InlineImage(doc, image_descriptor='res_y.png', width=Mm(150), height=Mm(100))
        im_res_sum = InlineImage(doc, image_descriptor='res_sum.png', width=Mm(150), height=Mm(100))
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%y - %H:%M")
        short_st = datetime.datetime.fromtimestamp(ts).strftime("%d_%m_%M%H")

        res = self.res

        context = {'date': st,
                   'bois': 'C24',
                   'var': 30,
                   'Image':
                       {'load': im_load,
                        'res_x': im_res_x,
                        'res_y': im_res_y,
                        'res_sum': im_res_sum
                        },
                   'res': res
                   }

        doc.render(context)
        doc.save("Rapport_" + short_st + ".docx")
        return print("Rapport genéré avec succès")

if __name__ == "__main__":
    test_3d()

'''
TODO : 
    [x] arrondi en notation scientifique en python
    [x] visuel charge répartie
    [] bien gérer la génération d'une charge répartie et d'une charge ponctuelle
    [] sortie format json ou dictionnaire ?
    [] nettoyage du code 
    [] ajouter des docstrings
'''
