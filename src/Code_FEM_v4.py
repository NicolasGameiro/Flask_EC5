# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:41:46 2022

@author: ngameiro
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
#%matplotlib notebook
from matplotlib.animation import FuncAnimation
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import time, datetime

class Mesh : 
    def __init__(self, dim, node_list = [], element_list = [], S_list = [], I_list = [], h = 0.22, b = 0.10, debug = False) :
        self.dim = dim
        self.node_list = np.array(node_list)
        self.element_list = np.array(element_list)
        self.S_list = np.array(S_list)
        self.I_list = np.array(I_list)
        self.S = h*b
        self.Iy = b*h**3/12
        self.Iz = h*b**3/12
        self.debug = debug
    
    def add_node(self,node) : 
        if len(node) != self.dim : 
            print("Error : uncorrect node format")
        else :
            found, index = self.check_node(node)
            if found == False : 
                self.node_list = np.append(self.node_list,[node], axis=0)
                print("noeud added")
            else :
                print("noeud not added")
        if self.debug == True : 
            print(self.node_list)
                
    def check_node(self,node) : 
        index = -1
        found = False
        while (found is not True) and (index+1 < len(self.node_list)) : 
            index += 1
            if (self.node_list[index][0] == node[0]) and (self.node_list[index][1] == node[1]) :
                found = True
        return found, index
        
    
    def del_node(self,node) : 
        if len(node) != self.dim : 
            print("Error : uncorrect node format")
        else :
            found, index = self.check_node(node)
            if found == True :
                self.node_list = np.delete(self.node_list, index , 0)
                print("noeud deleted")
            else : 
                print("node not found")            
            if self.debug == True : 
                print(self.node_list)
                
    def reset_node(self) : 
        self.node_list = np.array([])
        print("liste des noeuds cleared")
        if self.debug == True : 
            print(self.node_list)
        return
    
    def check_elem(self,elem) : 
        index = -1
        found = False
        while (found is not True) and (index+1 < len(self.element_list)) : 
            index += 1
            if (self.element_list[index][0] == elem[0]) and (self.element_list[index][1] == elem[1]) :
                found = True
        return found, index
    
    def add_element(self,elem) : 
        if len(elem) != self.dim : 
            print("Error : uncorrect node format")
        else :
            found, index = self.check_elem(elem)
            if found == False : 
                self.element_list = np.append(self.element_list,[elem], axis=0)
                print("element added")
            else :
                print("element not added")
            if self.debug == True : 
                print(self.element_list)
    
    def del_element(self, element) : 
        if len(element) != 2 : 
            print("Error : uncorrect element format")
        else :
            found, index = self.check_elem(element)
            if found == True :
                self.element_list = np.delete(self.element_list, index , 0)
                print("element deleted")
            else : 
                print("element not found")   
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
    
    def geom(self, pic = False) : 
        if self.dim == 2 :
            fig = self.geom2D(pic)
        else : 
            fig = self.geom3D(pic)
        return fig
    
    def geom2D(self, pic = False) : 
        fig = plt.figure(figsize=(8,6))
        x = [x for x in self.node_list[:,0]]
        y = [y for y in self.node_list[:,1]]
        size = 200
        offset = size/40000.
        plt.scatter(x, y, c='y', s=size, zorder=5)
        for i, location in enumerate(zip(x,y)):
            plt.annotate(i+1, (location[0]-offset, location[1]-offset), zorder=10)
        for i in range(len(self.element_list)) :
            xi,xj = self.node_list[self.element_list[i,0]-1,0],self.node_list[self.element_list[i,1]-1,0]
            yi,yj = self.node_list[self.element_list[i,0]-1,1],self.node_list[self.element_list[i,1]-1,1]
            plt.plot([xi,xj],[yi,yj],color = 'k', lw = 1, linestyle = '--')
        plt.axis('equal')
        plt.grid()
        if pic : 
            plt.savefig('static/images/geom.png', format='png' , dpi=200)
        return fig
    
    def geom3D(self, pic = False) : 
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
            plt.savefig('geom.png', format='png', dpi=200)
        return fig
        
class FEM_Model() : 
    def __init__(self, mesh, E = 2.1E11) :
        self.mesh = mesh
        self.E = E
        self.load = np.zeros([len(self.mesh.node_list),3])
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
        Rotation_matrix =  np.array([[c, s , 0, 0 , 0, 0],
                                     [-s, c, 0, 0 , 0, 0],
                                     [0, 0, 1, 0, 0 , 0],
                                     [0, 0 , 0 ,c ,s , 0 ],
                                     [0, 0, 0, -s, c, 0],
                                     [0, 0, 0, 0, 0 , 1]])
        return Rotation_matrix
    
    def mini_rot(self,c,s) : 
        R = np.array([[c, s],
                      [-s, c]])
        return R
    
    def K_elem(self,L_e) :
        S = self.mesh.S
        I = self.mesh.Iy
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
            self.sig[i,0] = self.load[i,0]/S/1e6 # traction/compression
            self.sig[i,1] = self.load[i,1]/S/1e6 # cisaillement
            self.sig[i,2] = self.load[i,2]/I*(h/2)/1e6 # flexion
        print(self.sig)
        
    
    def K_elem_3d(self,L,E,S,Iy,Iz,G,J) :
        Ktc = E*S/L
        KT = G*J/L
        Kf1 =  12*E*Iy/(L**3)
        Kf2 =  12*E*Iz/(L**3)
        Kf3 = 1
        Kf4 = 1
        K_elem = np.array([[Ktc, 0, 0, 0, 0, 0, -Ktc, 0, 0, 0, 0, 0],
                            [0, Kf1 , 0, 0, Kf2, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [-Ktc, 0, 0 , S, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
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
        

    def assemblage_2D(self) :
        BB = self.changement_coord()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL)*3,len(NL)*3])
        for i in range(len(EL)) :
            element = EL[i]
            L_e = self.get_length(element)
            c,s = self.get_angle(element)
            rot = self.Rot(c,s)
            # rotation matrice elem
            K_rot = rot.dot(self.K_elem(L_e)).dot(np.transpose(rot))
            M_global = M_global + self.changement_base(BB[i],K_rot)
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
    
    def plot_forces(self, pic = False) :
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
            plt.plot([xi,xj],[yi,yj],color = 'k', lw = 1, linestyle = '--')
        plt.quiver(NL[:,0], NL[:,1], F[:,0], F[:,1], color='r', angles='xy', scale_units='xy', scale=scale_force)
        plt.grid()
        plt.ylim([-1,4])
        plt.xlim([-1,5])
        #plt.axis('equal')
        #plt.show()
        if pic : 
            plt.savefig('static/images/load.png', format='png', dpi=200)
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
    
    def get_bc(self) : 
        """Return the boundary condition in a matrix format
        
        :return: matrix with 1 if the dof is blocked and 0 if the dof is free
        :rtype: np.array
        """
        BC = np.zeros(3*len(self.mesh.node_list))
        for i in self.lbc : 
            BC[i] = 1
        BC = BC.reshape((len(self.mesh.node_list),3))
        print(BC)
        return BC
    
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

    def plot_disp_f(self,scale=1e4,r=150,dir='x', pic = False) :
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
            plt.savefig('static/images/res_' + dir + '.png', format='png', dpi=200)
        return
        
    def __str__(self):
        return "fem solver"
    
    def U_table(self):
        tab = pt()
        if self.mesh.dim == 2 :
            tab.field_names = ["Node","Ux", "Uy", "Phi"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), self.U[i][0], self.U[i+1][0], self.U[i+2][0]])
        else : 
            tab.field_names = ["Node","Ux", "Uy", "Uz", "Phix", "Phiy", "Phiz"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), self.U[i][0], self.U[i+1][0], self.U[i+2][0], self.U[i+3][0], self.U[i+4][0], self.U[i+5][0]])
        print(tab)
        
    def R_table(self):
        tab = pt()
        if self.mesh.dim == 2 :
            tab.field_names = ["Node","Fx", "Fy", "Mz"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), self.React[i][0], self.React[i+1][0], self.React[i+2][0]])
        else : 
            tab.field_names = ["Node","Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i in range(len(self.mesh.node_list)) : 
                tab.add_row([int(i+1), self.React[i][0], self.React[i+1][0], self.React[i+2][0], self.React[i+3][0], self.React[i+4][0], self.React[i+5][0]])
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
    return

def test_2d() : 
    mesh = Mesh(2,[[0,0],[2,0]],[[1,2]],debug = True)
    mesh.add_node([4,0])
    mesh.del_node([5,0])
    mesh.add_node([4,0])
    mesh.add_node([2,3])
    mesh.add_element([2,3])
    mesh.add_element([3,4])
    mesh.add_element([4,1])
    mesh.add_element([4,2])
    #mesh.geom()
    #mesh.node_table()
    
    f = FEM_Model(mesh)
    f.apply_load([0,-1000,0],4)
    f.apply_bc([1,1,1],1)
    f.apply_bc([1,1,0],3)
    f.apply_distributed_load(1000, [1,2])
    f.apply_distributed_load(1000, [2,3])
    f.plot_forces(pic = True)
    f.solver_frame()
    U, React, res = f.get_res()
    f.plot_disp_f(dir='x', pic = True)
    f.plot_disp_f(dir='y' , pic = True)
    f.plot_disp_f(dir='sum', pic = True)
    f.plot_disp_f_ex()
    #f.U_table()
    #f.R_table()
    f.stress()
    #f.rapport()
    return 

def test_cantilever() : 
    mesh = Mesh(2,[[0,0],[3,0]],[[1,2]],debug=False)
    mesh.add_node([4,0])
    mesh.add_node([5,0])
    mesh.add_node([6,0])
    mesh.add_element([2,3])
    mesh.add_element([3,4])
    mesh.add_element([4,5])
    mesh.node_table()
    f = FEM_Model(mesh)
    f.apply_load([0,-1000,0],5)
    f.apply_bc([1,1,1],1)
    f.plot_forces()
    f.solver_frame()
    U, React = f.get_res()
    f.plot_disp_f_ex(scale=1e2)
    f.plot_disp_f(scale=1e2,dir='y')
    f.U_table()
    return 

if __name__ == "__main__" :
    test_2d()
    
"""
TODO : 
    [] arrondi en notation scientifique en python
    []
"""