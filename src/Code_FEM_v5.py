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

from prettytable import PrettyTable as pt
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.figsize"] = (8,6)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
from matplotlib.patches import Rectangle, Polygon
#%matplotlib notebook
from matplotlib.animation import FuncAnimation
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import time, datetime

class Mesh : 
    def __init__(self, dim, node_list = [], element_list = [], S_list = [], I_list = [], h = 22, b = 10, debug = False) :
        self.dim = dim
        self.node_list = np.empty((0,dim))
        self.element_list = np.empty((0,2),dtype = int)
        self.name = np.empty((0,1))
        self.color = np.empty((0,1))
        self.Section = np.empty((0,2))
        self.S_list = np.array(S_list)
        self.I_list = np.array(I_list)
        self.debug = debug
    
    def add_node(self,node) : 
        if len(node) != self.dim : 
            print("Erreur : format du noeud incorrect")
        else :
            found, index = self.check_node(node)
            if found == False : 
                self.node_list = np.append(self.node_list,np.array([node]), axis=0)
                print("noeud ajouté")
            else :
                print("noeud deja dans le maillage")
        if self.debug == True : 
            print(self.node_list)
                
    def check_node(self,node) : 
        index = -1
        found = False
        while (found is not True) and (index+1 < len(self.node_list)) and (self.node_list.size != 0) : 
            index += 1
            if (self.node_list[index][0] == node[0]) and (self.node_list[index][1] == node[1]) :
                found = True
        return found, index
        
    
    def del_node(self,node) : 
        if len(node) != self.dim : 
            print("Erreur : format du noeud incorrect")
        else :
            found, index = self.check_node(node)
            if found == True :
                self.node_list = np.delete(self.node_list, index , 0)
                print("noeud supprimé")
            else : 
                print("noeud non trouvé")            
            if self.debug == True : 
                print(self.node_list)
                
    def reset_node(self) : 
        self.node_list = np.array([])
        print("liste des noeuds vidée")
        if self.debug == True : 
            print(self.node_list)
        return
    
    ### GESTION DES ELEMENTS
    
    def check_elem(self,elem) : 
        index = -1
        found = False
        while (found is not True) and (index+1 < len(self.element_list)) and (self.element_list.size != 0): 
            index += 1
            if (self.element_list[index][0] == elem[0]) and (self.element_list[index][1] == elem[1]) :
                found = True
        return found, index
    
    def add_element(self, elem, name = "poutre", color = "k", h = "22", l = "10") : 
        if len(elem) != self.dim : 
            print("Erreur : format de l'element incorrect")
        else :
            found, index = self.check_elem(elem)
            if found == False : 
                self.element_list = np.append(self.element_list,np.array([elem]), axis=0)
                self.name = np.append(self.name, np.array(name))
                self.color = np.append(self.color, np.array(color))
                self.Section = np.append(self.Section, np.array([[h, l]]), axis = 0)
                print("element ajouté")
            else :
                print("element deja dans le maillage")
            if self.debug == True : 
                print(self.element_list)
                print(self.name)
                print(self.color)
                print(self.Section)
    
    def del_element(self, element) : 
        if len(element) != self.dim : 
            print("Erreur : format de l'element incorrect")
        else :
            found, index = self.check_elem(element)
            if found == True :
                self.element_list = np.delete(self.element_list, index , 0)
                print("element supprimé")
            else : 
                print("element non trouvé")   
            if self.debug == True : 
                print(self.element_list)
                
    def add_section(self,S) : 
        self.S_list = np.append(self.S_list,[S], axis=0)
        
    def node_table(self):
        tab = pt()
        if self.dim == 2 :
            tab.field_names = ["Node","X", "Y"]
            for i in range(len(self.node_list)) : 
                tab.add_row([int(i+1), self.node_list[i,0], self.node_list[i,1]])
        else : 
            tab.field_names = ["Node","X", "Y", "Z"]
            for i in range(len(self.node_list)) : 
                tab.add_row([int(i+1), self.node_list[i,0], self.node_list[i,1], self.node_list[i,2]])
        print(tab)
    
    def __str__(self):
        return f""" Information sur le maillage : \n
    - Nombre de noeuds : {len(self.node_list)}\n 
    - Nombre d'éléments : {len(self.element_list)}
    """
    
    def geom(self, pic = False, path = "./") : 
        if self.dim == 2 :
            fig = self.geom2D(pic)
        else : 
            fig = self.geom3D(pic)
        return fig
    
    def geom2D(self, pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        x = [x for x in self.node_list[:, 0]]
        y = [y for y in self.node_list[:, 1]]
        size = 10
        offset = size / 40000.
        ax.scatter(x, y, c='k', marker="s", s=size, zorder=5)
        color_list = []
        for i, location in enumerate(zip(x, y)):
            ax.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)
        for i in range(len(self.element_list)):
            xi, xj = self.node_list[self.element_list[i, 0] - 1, 0], self.node_list[self.element_list[i, 1] - 1, 0]
            yi, yj = self.node_list[self.element_list[i, 0] - 1, 1], self.node_list[self.element_list[i, 1] - 1, 1]
            ax.plot([xi, xj], [yi, yj], color=self.color[i], lw=2, linestyle='--',
            label=self.name[i] if self.color[i] not in color_list else '')

            h = self.Section[i][0]/100

            if xi != xj :
                pt1 = [xi , yi - h/2]
                pt2 = [xj , yj-h/2]
                pt3 = [xj , yj + h/2]
                pt4 = [xi, yi + h/2]
            else :
                pt1 = [xi - h/2 , yi]
                pt2 = [xj -h/2 , yj]
                pt3 = [xj + h/2, yj ]
                pt4 = [xi + h/2, yi ]

            x = pt1[0], pt2[0], pt3[0], pt4[0], pt1[0]
            y = pt1[1], pt2[1], pt3[1], pt4[1], pt1[1]
            ax.add_patch(Polygon(xy=list(zip(x, y)), color=self.color[i], fill=True, alpha=0.3, lw=0))
            # pour verifier que la legende n'existe pas deja
            if (self.color == self.color[i]).sum() > 1:
                color_list.append(self.color[i])
        ax.axis('equal')
        ax.legend()
        plt.grid()
        if pic:
            plt.savefig(path + 'geom.png', format='png', dpi=200)
        return fig
    
    def geom3D(self, pic = False, path = "./") : 
        fig = plt.figure(figsize=(8,6))
        #plt.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        x = [x for x in self.node_list[:,0]]
        y = [y for y in self.node_list[:,1]]
        z = [z for z in self.node_list[:,2]]
        ax.scatter(x, y, z, c='y', s=200, zorder=1)
        for i, location in enumerate(zip(x,y)):
            ax.text(x[i],y[i],z[i],str(i+1),size=20,zorder=2,color = "k")
        for i in range(len(self.element_list)) :
            xi,xj = self.node_list[self.element_list[i,0]-1,0],self.node_list[self.element_list[i,1]-1,0]
            yi,yj = self.node_list[self.element_list[i,0]-1,1],self.node_list[self.element_list[i,1]-1,1]
            zi,zj = self.node_list[self.element_list[i,0]-1,2],self.node_list[self.element_list[i,1]-1,2]
            line, = ax.plot([xi,xj],[yi,yj],[zi,zj],color = 'k', lw = 1, linestyle = '--')
            line.set_label('undeformed')
        ax.set_title("Structure")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        if pic : 
            plt.savefig(path + 'geom.png', format='png', dpi=200)
        return fig
        
class FEM_Model() : 
    def __init__(self, mesh, E = 10.0E9) :
        self.mesh = mesh
        self.E = E
        self.load = np.zeros([len(self.mesh.node_list),3])
        self.dist_load = np.array([[1,2,0]])
        self.bc = np.eye(len(self.mesh.node_list)*3)
        self.U = np.zeros(len(self.mesh.node_list)*3)
        self.React = np.zeros(len(self.mesh.node_list)*3)
        self.lbc = []
    
    def test(self) :
        self.mesh.geom()
        
    def apply_load(self,node_load,node):
        if len(node_load) != 3 : 
            print("Error : uncorrect load format (must be 3 elements Fx, Fy and Mz)")
        elif node > len(self.mesh.node_list) :
            print("Error : node specified not in the mesh")
        else :
            self.load[node-1,:] = node_load
            print("nodal load applied")
            if self.mesh.debug == True :
                print(self.load)
            
    def apply_distributed_load(self,q,element):
        L = self.get_length(element)
        Q = np.array([0,
                      -q*L/2,
                      -q*L**2/12,
                      0,
                      -q*L/2,
                      q*L**2/12])
        self.load[element[0]-1] = self.load[element[0]-1] + Q [:3]        
        self.load[element[1]-1] = self.load[element[1]-1] + Q [3:6]
        self.dist_load = np.append(self.dist_load, [[ element[0], 
                                                    element[1],
                                                    q ]], axis=0)
        #print(self.dist_load)
    
    def apply_bc(self,node_bc,node):
        if len(node_bc) != 3 : 
            print("Error : uncorrect bc format (must be 3 elements Fx, Fy and Mz)")
        elif node > len(self.mesh.node_list) :
            print("Error : node specified not in the mesh")
        else :
            for i in range(len(node_bc)) : 
                if node_bc[i] == 1 : 
                    self.lbc.append(i+3*(node-1))
            print("boundary condition applied")
    
    def Rot(self,c,s) : 
        Rotation_matrix =  np.array([[c, -s , 0, 0 , 0, 0],
                                     [s, c, 0, 0 , 0, 0],
                                     [0, 0, 1, 0, 0 , 0],
                                     [0, 0 , 0 ,c ,-s , 0 ],
                                     [0, 0, 0, s, c, 0],
                                     [0, 0, 0, 0, 0 , 1]])
        return Rotation_matrix
    
    def mini_rot(self,c,s) : 
        R = np.array([[c, s],
                      [-s, c]])
        return R
    
    def K_elem(self,L_e, h, b) :
        S = h*b*1e-4
        I = b*h**3/12*1e-8
        K_elem = self.E/L_e*np.array([[S, 0, 0 , -S, 0, 0],
                                    [0, 12*I/L_e**2 , 6*I/L_e, 0, -12*I/L_e**2, 6*I/L_e],
                                    [0, 6*I/L_e , 4*I, 0, -6*I/L_e, 2*I],
                                    [-S, 0, 0 , S, 0, 0],
                                    [0, -12*I/L_e**2 , -6*I/L_e, 0, 12*I/L_e**2, -6*I/L_e],
                                    [0, 6*I/L_e , 2*I, 0, -6*I/L_e, 4*I]])
        return K_elem
    
    def stress(self) : 
        S = self.mesh.S
        I = self.mesh.Iy
        h = 0.22
        self.sig = np.zeros([len(self.mesh.node_list),3])
        for i in range(len(self.mesh.node_list)) : 
            #en MPa
            self.sig[i,0] = self.load[i,0]/S/1e6 # traction/compression (en MPa)
            self.sig[i,1] = self.load[i,1]/S/1e6 # cisaillement (en MPa)
            self.sig[i,2] = self.load[i,2]/I*(h/2)/1e6 # flexion (en MPa)
        print(self.sig)
        
    
    def K_elem_3d(self, L : float , E : float , S : float , Iy : float , Iz : float , G : float , J : float , ay : float = 0, az : float = 0) -> np.array :
        """ Calcul de la matrice de raideur avec prise en compte de l'énergie cisaillement avec les termes ay et az.
    
        :param L: longueur de l'element
        :type L: float
        :param E: Module d'Young
        :type E: float
        :param S: Section
        :type S: float
        :param Iy: Inertie
        :type Iy: float
        :param Iz: Inertie
        :type Iz: float
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
        Ktc = E * S / L
        KT = G * J / L
        Kf1 = 12 * E * Iz / (L ** 3 * (1 + az))
        Kf2 = 12 * E * Iy / (L ** 3 * (1 + ay))
        Kf3 = -6 * E * Iy / (L ** 2 * (1 + ay))
        Kf4 = 6 * E * Iz / (L ** 2 * (1 + az))
        Kf5 = (4 + ay) * E * Iy / (L * (1 + ay))
        Kf6 = (4 + az) * E * Iz / (L * (1 + az))
        K_elem = np.array([[Ktc, 0, 0, 0, 0, 0, -Ktc, 0, 0, 0, 0, 0], #1
                           [0, Kf1, 0, 0, 0, Kf4, 0, -Kf1, 0, 0, 0, Kf4],
                           [0, 0, Kf2, 0, Kf3, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, KT, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, Kf5, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, Kf6, 0, 0, 0, 0, 0, 0],
                           [-Ktc, 0, 0, 0, 0, 0, Ktc, 0, 0, 0, 0, 0, 0], #7
                           [0, 0, 0, 0, 0, 0, 0, Kf1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, Kf2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, KT, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Kf5, 0],
                           [0, Kf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, Kf6]])
        return K_elem
        
    def changement_base(self,P,M) : 
        return P.dot(M).dot(np.transpose(P))

    def changement_coord(self) :
        BB = []
        for i in range(len(self.mesh.element_list)) : # Une matrice de changement de coord par element
            #print("generation de la matrice de passage de l'element ", i + 1, ":")
            B = np.zeros([len(self.mesh.node_list)*3,6])
            noeud1 = self.mesh.element_list[i,0]
            noeud2 = self.mesh.element_list[i,1]
            B[(noeud1 - 1)*3, 0] = 1
            B[(noeud1 - 1)*3 + 1 , 1] = 1
            B[(noeud1 - 1)*3 + 2, 2] = 1
            B[(noeud2 - 1)*3 , 3] = 1
            B[(noeud2 - 1)*3 + 1, 4] = 1
            B[(noeud2 - 1)*3 + 2, 5] = 1
            BB.append(B)
        return BB
    
    def get_length(self,element) : 
        noeud1 = element[0]
        noeud2 = element[1]
        x_1 = self.mesh.node_list[noeud1-1,0]
        x_2 = self.mesh.node_list[noeud2-1,0]
        y_1 = self.mesh.node_list[noeud1-1,1]
        y_2 = self.mesh.node_list[noeud2-1,1]
        L_e = np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2)
        return L_e
    
    def get_angle(self,element) : 
        """ Return the cosinus and the sinus associated with the angle of the element
        in the global coordinate

        :return: tuple with cosinus and sinus
        :rtype: 2-uple
        """
        noeud1 = element[0]
        noeud2 = element[1]
        x_1 = self.mesh.node_list[noeud1-1,0]
        x_2 = self.mesh.node_list[noeud2-1,0]
        y_1 = self.mesh.node_list[noeud1-1,1]
        y_2 = self.mesh.node_list[noeud2-1,1]
        L_e = np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2)
        c = (x_2-x_1)/L_e
        s = (y_2-y_1)/L_e
        return c,s
    
    def get_bc(self) : 
        """Return the boundary condition in a matrix format
        
        :return: matrix with 1 if the dof is blocked and 0 if the dof is free
        :rtype: np.array
        """
        BC = np.zeros(3*len(self.mesh.node_list))
        for i in self.lbc : 
            BC[i] = 1
        BC = BC.reshape((len(self.mesh.node_list),3))
        return BC
        

    def assemblage_2D(self) :
        """ Return the global stiffness matrix of the mesh
        
        :return: matrix of size(dll*3*nb_node,dll*3*nb_node)
        :rtype: np.array

        """
        BB = self.changement_coord()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL)*3,len(NL)*3])
        for i in range(len(EL)) :
            element = EL[i]
            L_e = self.get_length(element)
            c,s = self.get_angle(element)
            rot = self.Rot(c,s)
            h, b = self.mesh.Section[i,0], self.mesh.Section[i,1]
            # rotation matrice elem
            K_rot = rot.dot(self.K_elem(L_e, h , b)).dot(np.transpose(rot))
            M_global = M_global + self.changement_base(BB[i],K_rot)
            if self.mesh.debug == True : 
                print("element " + str(i+1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.K_elem(L_e, h, b))
                print(K_rot)
        return M_global

    def solver_frame(self) :
        self.bc = np.delete(self.bc,self.lbc,axis=1)
        K_glob = self.assemblage_2D()
        K_glob_r = np.transpose(self.bc).dot(K_glob).dot(self.bc)
        ### en cas de matrice singuliaire 
        m = 0
        K_glob_r = K_glob_r + np.eye(K_glob_r.shape[1])*m
        ###
        F = np.vstack(self.load.flatten())
        F_r = np.transpose(self.bc).dot(F)
        U_r = inv(K_glob_r).dot(F_r)
        self.U = self.bc.dot(U_r)
        self.React = K_glob.dot(self.U) - F
    
    def get_res(self):
        self.res = {}
        self.res['U'] = []
        self.res['React'] = []
        self.res['node'] = []
        self.res['elem'] = []
        for i in range(len(self.mesh.node_list)) : 
            self.res['U'].append({'node' : i + 1 , 'Ux' : self.U[i][0] , 'Uy' : self.U[i+1][0] , 'phi' : self.U[i+2][0]})
            self.res['React'].append({'node' : i + 1 , 'Fx' : self.React[i][0] , 'Fy' : self.React[i+1][0] , 'Mz' : self.React[i+2][0]})
            self.res['node'].append({'node' : i + 1 , 'X' : self.mesh.node_list[i][0] , 'Y' : self.mesh.node_list[i][1]})
            self.res['elem'].append({'elem' : i + 1 , 'node i' : self.mesh.element_list[i][0] , 'node j' : self.mesh.element_list[i][1]})
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
                     zorder = 6)
        plt.plot([pt1[0], pt2[0]], [pt1[1] + 1, pt2[1] + 1], lw=1, color='r', zorder = 6)
        ax.text(x1 + dx/2*0.9 , y1 + dy/2 + amplitude*1.2,
                "q = " + str(q/1000) + " kN/m",
                size=10, zorder=2, color="k")
        x = [pt1[0], pt2[0], pt2[0], pt1[0], pt1[0]]
        y = [pt1[1], pt2[1], pt2[1]+1, pt1[1]+1, pt1[1]]
    
        ax.add_patch(Polygon(xy=list(zip(x, y)), fill=True, color='red', alpha=0.1, lw=0))
        return
    
    def charge_3D(self, pt1, pt2, q):
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        z1, z2 = pt1[2], pt2[2]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        #a = np.arctan(dy / dx)
        nb_pt = 5
        amplitude = 1
        x = np.linspace(x1, x2, nb_pt)
        y = np.linspace(y1, y2, nb_pt)
        z = np.linspace(z1, z2, nb_pt)
    
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    
        for i in range(0, nb_pt):
            a = Arrow3D([x[i], x[i]], 
                        [y[i], y[i]], 
                        [z[i] + amplitude , z[i]], 
                        mutation_scale=10, 
                        lw=2, arrowstyle="-|>", color="r")
            ax.add_artist(a)
        line, = ax.plot([x1, x2 ], [y1, y2], [z1 + amplitude, z2 + amplitude], color='r', lw=1, linestyle='--')
        line.set_label('undeformed')
        ax.text(x1 + dx/2, y1 + dy/2, z1 + dz/2,
                "q = " + str(q) + " kN/m",
                size=20, zorder=2, color="k")
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 + amplitude
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show()
        return
    
    def plot_forces(self, type = 'nodal', pic = False, path = "./") :
        plt.figure()
        F = self.load
        NL = self.mesh.node_list
        scale_force = np.max(np.abs(F))
        x = [x for x in self.mesh.node_list[:,0]]
        y = [y for y in self.mesh.node_list[:,1]]
        size = 200
        offset = size/40000.
        plt.scatter(x, y, c='y', s=size, zorder=5)
        for i, location in enumerate(zip(x,y)):
            plt.annotate(i+1, (location[0]-offset, location[1]-offset), zorder=10)
        for i in range(len(self.mesh.element_list)) :
            xi,xj = self.mesh.node_list[self.mesh.element_list[i,0]-1,0],self.mesh.node_list[self.mesh.element_list[i,1]-1,0]
            yi,yj = self.mesh.node_list[self.mesh.element_list[i,0]-1,1],self.mesh.node_list[self.mesh.element_list[i,1]-1,1]
            plt.plot([xi,xj],[yi,yj],color = self.mesh.color[i], lw = 1, linestyle = '--')
        ### Trace les efforts
        if type == 'nodal':
            plt.quiver(NL[:,0] - F[:,0]/scale_force , NL[:,1] - F[:,1]/scale_force , F[:,0], F[:,1], color='r', angles='xy', scale_units='xy', scale=scale_force)
        elif type == 'dist':
            for elem in self.dist_load[1:]:
                pt1 = self.mesh.node_list[elem[0]-1]
                pt2 = self.mesh.node_list[elem[1]-1]   
                self.charge_2D(pt1,pt2,elem[2])
        plt.grid()
        plt.ylim([-1,max(x)])
        plt.xlim([-1,max(y)])
        plt.axis('equal')
        #plt.show()
        if pic : 
            plt.savefig(path + 'load.png', format='png', dpi=200)
        return
    
    def interpol(self,x1,x2,y1,y2,y3,y4,r) : 
        x3 = x1
        x4 = x2
        V = np.array([[1, x1, x1**2, x1**3],
                      [1, x2, x2**2, x2**3],
                      [0,1, 2*x3, 3*x3**2],
                      [0,1, 2*x4, 3*x4**2]])
        #print(V)
        R = np.array([y1,y2,y3,y4])
        R = np.vstack(R)
        P = np.hstack(inv(V).dot(R))
        P = P[::-1]
        p = np.poly1d([x for x in P])
        x = np.linspace(x1,x2,r)
        y = p(x)
        return x,y
    
    def plot_disp_f_ex(self,scale=1e4,r=150) :
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        for i in range(len(EL)) :
            xi,xj = NL[EL[i,0]-1,0],NL[EL[i,1]-1,0]
            yi,yj = NL[EL[i,0]-1,1],NL[EL[i,1]-1,1]
            plt.plot([xi,xj],[yi,yj],color = 'k', lw = 1, linestyle = '--')
        for i in range(len(EL)) :
            x1 = NL[EL[i,0]-1,0]+U[(EL[i,0]-1)*3]*scale
            x2 = NL[EL[i,1]-1,0]+U[(EL[i,1]-1)*3]*scale
            y1 = NL[EL[i,0]-1,1]+U[(EL[i,0]-1)*3+1]*scale
            y2 = NL[EL[i,1]-1,1]+U[(EL[i,1]-1)*3+1]*scale
            y3 = U[(EL[i,0]-1)*3+2]
            y4 = U[(EL[i,1]-1)*3+2]
            L_e = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            c = np.round((x2-x1)/L_e,2)
            #print("c =", c)
            a = np.arccos(c)%1
            #print("a = ", a)
            x,y = self.interpol(x1[0],x2[0],y1[0],y2[0],y3[0] + a,-y4[0] + a,r)
            x_scatter.append(x)
            y_scatter.append(y)
            color.append(np.linspace(U[(EL[i,0]-1)*3+1],U[(EL[i,1]-1)*3+1],r))
        #Permet de reverse la barre de couleur si max negatif 
        if min(U) > 0 :
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0 : 
            cmap = plt.get_cmap('jet_r')
        plt.scatter(x_scatter,y_scatter,c = color,cmap = cmap,s=10, edgecolor = 'none' )
        plt.colorbar(label='disp'
                     , orientation='vertical') #ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        return

    def plot_disp_f(self,scale=1e3,r=150,dir='x', pic = False, path = "./") :
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        for i in range(len(EL)) :
            xi,xj = NL[EL[i,0]-1,0],NL[EL[i,1]-1,0]
            yi,yj = NL[EL[i,0]-1,1],NL[EL[i,1]-1,1]
            plt.plot([xi,xj],[yi,yj],color = 'k', lw = 1, linestyle = '--')
        for i in range(len(EL)) :
            if dir == 'y' : 
                plt.title("y")
                x_scatter.append(np.linspace(NL[EL[i,0]-1,0],NL[EL[i,1]-1,0],r))
                y_scatter.append(np.linspace(NL[EL[i,0]-1,1]+U[(EL[i,0]-1)*3+1]*scale,NL[EL[i,1]-1,1]+U[(EL[i,1]-1)*3+1]*scale,r))
                color.append(np.linspace(U[(EL[i,0]-1)*3+1],U[(EL[i,1]-1)*3+1],r))
            elif dir == "x" : 
                plt.title("x")
                x_scatter.append(np.linspace(NL[EL[i,0]-1,0]+U[(EL[i,0]-1)*3]*scale,NL[EL[i,1]-1,0]+U[(EL[i,1]-1)*3]*scale,r))
                y_scatter.append(np.linspace(NL[EL[i,0]-1,1],NL[EL[i,1]-1,1],r))
                color.append(np.linspace(U[(EL[i,0]-1)*3],U[(EL[i,1]-1)*3],r))
            elif dir == "sum" : 
                plt.title("sum")
                x_scatter.append(np.linspace(NL[EL[i,0]-1,0]+U[(EL[i,0]-1)*3]*scale,NL[EL[i,1]-1,0]+U[(EL[i,1]-1)*3]*scale,r))
                y_scatter.append(np.linspace(NL[EL[i,0]-1,1]+U[(EL[i,0]-1)*3+1]*scale,NL[EL[i,1]-1,1]+U[(EL[i,1]-1)*3+1]*scale,r))
                color.append(np.linspace(U[(EL[i,0]-1)*3]+U[(EL[i,0]-1)*3+1],U[(EL[i,1]-1)*3]+U[(EL[i,1]-1)*3+1],r))
        #Permet de reverse la barre de couleur si max negatif 
        if min(U) > 0 :
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0 : 
            cmap = plt.get_cmap('jet_r')
        plt.scatter(x_scatter,y_scatter,c = color,cmap = cmap,s=10, edgecolor = 'none' )
        plt.colorbar(label='disp'
                     , orientation='vertical') #ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        if pic : 
            plt.savefig(path + 'res_' + dir + '.png', format='png', dpi=200)
        return
        
    def __str__(self):
        return "fem solver"
    
    def U_table(self):
        tab = pt()
        if self.mesh.dim == 2 :
            tab.field_names = ["Node","Ux (m)", "Uy (m)", "Phi (rad)"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1),
                             np.format_float_scientific(self.U[i*3], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i*3+1], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i*3+2], precision = 2, exp_digits=2)])
        else :
            tab.field_names = ["Node","Ux (m)", "Uy (m)", "Uz (m)", "Phix (rad)", "Phiy (rad)", "Phiz (rad)"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), 
                             np.format_float_scientific(self.U[i][0], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i][1], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i][2], precision = 2, exp_digits=2), 
                             np.format_float_scientific(self.U[i][3], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i][4], precision = 2, exp_digits=2),
                            np.format_float_scientific(self.U[i][5], precision = 2, exp_digits=2)])
        print(tab)
        
    def R_table(self):
        tab = pt()
        if self.mesh.dim == 2 :
            tab.field_names = ["Node","Fx (N)", "Fy (N)", "Mz (N.m)"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), 
                             np.format_float_scientific(self.React[i][0], precision = 2, exp_digits=2),
                               np.format_float_scientific(self.React[i+1][0], precision = 2, exp_digits=2),
                               np.format_float_scientific(self.React[i+2][0], precision = 2, exp_digits=2)])
        else : 
            tab.field_names = ["Node","Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1),
                             np.format_float_scientific(self.React[i][0], precision = 2, exp_digits=2),
                             np.format_float_scientific(self.React[i+1][0], precision = 2, exp_digits=2),
                             np.format_float_scientific(self.React[i+2][0], precision = 2, exp_digits=2),
                             np.format_float_scientific(self.React[i+3][0], precision = 2, exp_digits=2),
                             np.format_float_scientific(self.React[i+4][0], precision = 2, exp_digits=2),
                             np.format_float_scientific(self.React[i+5][0], precision = 2, exp_digits=2)])
        print(tab)
    
    def rapport(self) : 
        doc = DocxTemplate("cctr_template.docx")
        
        im_load = InlineImage(doc, image_descriptor='load.png', width=Mm(150), height=Mm(100))
        im_res_x = InlineImage(doc, image_descriptor='res_x.png', width=Mm(150), height=Mm(100))
        im_res_y = InlineImage(doc, image_descriptor='res_y.png', width=Mm(150), height=Mm(100))
        im_res_sum = InlineImage(doc, image_descriptor='res_sum.png', width=Mm(150), height=Mm(100))
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%y - %H:%M")
        short_st = datetime.datetime.fromtimestamp(ts).strftime("%d_%m_%M%H")
        
        res = self.res
        
        context = { 'date' : st,
                   'bois' :'C24',
                   'var' : 30,
                   'Image' : 
                       { 'load' : im_load , 
                        'res_x' : im_res_x ,
                        'res_y' : im_res_y ,
                        'res_sum' : im_res_sum 
                        } ,
                   'res' : res
                   }
        
        doc.render(context)
        doc.save("Rapport_" + short_st + ".docx")
        return print("Rapport genéré avec succès")

def test_3d() :
    m1 = Mesh(3,[[0,0,0]],[[1,2]])
    m1.add_node([1,0,0])
    m1.add_node([1,1,0])
    m1.add_node([1,1,1])
    m1.add_element([2,3])
    m1.add_element([3,4])
    m1.add_element([4,1])
    print(m1)
    m1.geom()
    m1.node_table()
    f = FEM_Model(m1)
    f.charge_3D([0,0,0],[1,1,1],5)
    return

def validation_2d() : 
    mesh = Mesh(2,[],[],debug = True)
    mesh.add_node([0,0])
    mesh.add_node([0,2])
    mesh.add_node([2,2])
    mesh.add_element([1,2], "barre", "b",12, 12)
    mesh.add_element([2,3], "barre", "b",12, 12)
    mesh.geom()
    
    f = FEM_Model(mesh)
    f.apply_load([0,-100,0],3)
    f.apply_bc([1,1,1],1)
    print(f.get_bc())
    f.plot_forces(type = 'nodal', pic = True)
    f.solver_frame()
    f.plot_disp_f(dir='x', pic = True)
    f.plot_disp_f(dir='y' , pic = True)
    f.plot_disp_f(dir='sum', pic = True)
    f.plot_disp_f_ex()
    f.U_table()
    f.R_table()
    
def validation_2d() : 
    mesh = Mesh(2,[],[],debug = True)
    mesh.add_node([0,0])
    mesh.add_node([0,80]) #inches
    mesh.add_node([100,80]) #inches
    mesh.add_element([1,2], "barre", "b",12, 12)
    mesh.add_element([2,3], "barre", "b",12, 12)
    #mesh.geom()
    
    f = FEM_Model(mesh)
    f.apply_distributed_load(100, [2,3])
    f.apply_bc([1,1,1],1)
    f.apply_bc([0,1,0],3)
    print(f.get_bc())
    f.plot_forces(type = 'nodal', pic = True)
    f.solver_frame()
    f.plot_disp_f(dir='x', pic = True)
    f.plot_disp_f(dir='y' , pic = True)
    f.plot_disp_f(dir='sum', pic = True)
    #f.plot_disp_f_ex()
    f.U_table()
    f.R_table()


def test_2d() : 
    mesh = Mesh(2,[],[],debug = False)
    p = 6.5
    h = 2.5
    mesh.add_node([0,0])
    mesh.add_node([p/2,0])
    mesh.add_node([p,0])
    mesh.add_node([p/2,h])
    mesh.add_node([p/4,h/2])
    mesh.add_node([3*p/4,h/2])
    mesh.add_element([1,2], "entrait", "r", 22, 10)
    mesh.add_element([2,3], "entrait", "r", 22 , 10)
    mesh.add_element([3,6], "arba", "g", 20, 8)
    mesh.add_element([6,4], "arba", "g", 20, 8)
    mesh.add_element([4,5], "arba", "g", 20, 8)
    mesh.add_element([5,1], "arba", "g", 20, 8)
    mesh.add_element([4,2], "poinçon", "b", 10, 10)
    mesh.add_element([2,5], "jdf", "m", 10,10)
    mesh.add_element([2,6], "jdf", "m", 10,10)
    mesh.geom()
    #mesh.node_table()
    
    f = FEM_Model(mesh)
    #f.apply_load([0,-1000,0],4)
    f.apply_bc([1,1,1],1)
    f.apply_bc([1,1,1],3)
    print(f.get_bc())
    f.apply_distributed_load(2000, [1,4])
    f.apply_distributed_load(2000, [4,3])
    f.plot_forces(type = 'dist', pic = True)
    f.solver_frame()
    U, React, res = f.get_res()
    f.plot_disp_f(dir='x', scale = 1e3, pic = True)
    f.plot_disp_f(dir='y' , scale = 1e3, pic = True)
    f.plot_disp_f(dir='sum', scale = 1e3, pic = True)
    #f.plot_disp_f_ex()
    f.U_table()
    f.R_table()
   # f.stress()
    #f.rapport()
    return 

def test_cantilever() : 
    mesh = Mesh(2,[],[],debug=False)
    mesh.add_node([0,0])
    mesh.add_node([1,0])
    mesh.add_node([2,0])
    mesh.add_node([4,0])
    mesh.add_node([5,0])
    mesh.add_element([1,2], "entrait", "r", 22, 10)
    mesh.add_element([2,3], "entrait", "r", 22, 10)
    mesh.add_element([3,4], "entrait", "r", 22, 10)
    mesh.add_element([4,5], "entrait", "r", 22, 10)
    mesh.node_table()
    f = FEM_Model(mesh)
    f.apply_load([0,-1000,0],5)
    f.apply_bc([1,1,1],1)
    f.plot_forces(type = 'dist', pic = False)
    f.solver_frame()
    U, React = f.get_res()
    #f.plot_disp_f_ex(scale=1e2)
    f.plot_disp_f(scale=1e2,dir='y')
    f.U_table()
    return 

if __name__ == "__main__" :
    test_2d()
    
'''
TODO : 
    [x] arrondi en notation scientifique en python
    [x] visuel charge répartie
    [] bien gérer la génération d'une charge répartie et d'une charge ponctuelle
    [] sortie format json ou dictionnaire ?
    [] nettoyage du code 
    [] ajouter des docstrings
'''