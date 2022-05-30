# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:42:45 2022

@author: ngameiro
"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

class maison(): 
    def __init__(self,ax,H,h,L,l,d) : 
        self.Hauteur = H
        self.hauteur_toiture = h
        self.Longueur = L
        self.largueur = l
        self.debord = d
        self.h_mur = self.Hauteur-self.hauteur_toiture
        self.ax = ax
        
    def maconnerie(self) : 
        #mur_1
        pt1 = (0,0,0) 
        pt2 = (self.Longueur, 0, 0)
        pt3 = (self.Longueur, 0, self.Hauteur-self.hauteur_toiture)
        pt4 = (0, 0, self.Hauteur-self.hauteur_toiture)
        mur_1 = [[pt1, pt2, pt3, pt4]]
        self.ax.add_collection3d(Poly3DCollection(mur_1,color='k',alpha= 0.3))
        
        #mur_2
        pt5 = (self.Longueur, self.largueur, self.Hauteur-self.hauteur_toiture)
        pt6 = (self.Longueur, self.largueur, 0)
        mur_2 = [[pt2, pt3, pt5, pt6]]
        self.ax.add_collection3d(Poly3DCollection(mur_2,color='k',alpha= 0.3))
        
        #mur_3
        pt7 = (0, self.largueur, 0)
        pt8 = (0, self.largueur, self.Hauteur-self.hauteur_toiture)
        mur_3 = [[pt5, pt6, pt7, pt8]]
        self.ax.add_collection3d(Poly3DCollection(mur_3,color='k',alpha= 0.3))
        
        #mur_4
        mur_4 = [[pt7, pt8, pt4, pt1]]
        self.ax.add_collection3d(Poly3DCollection(mur_4,color='k',alpha= 0.3))
        return print("murs créés")
    
    def toiture(self, type = "2_pentes"):
        if type == "2_pentes" : 
            self.toiture_2pents()
        elif type == "4_pentes":
            self.toiture_4pents()
        else :
            print("pas le bon type de toiture")
            
    def toiture_2pents(self) :
        """ plot une toiture 2 pents à partir de largeur, longueur et hauteur de la toiture.
        """
        pt1 = (0, -self.debord, self.h_mur)
        pt2 = (self.Longueur , -self.debord, self.h_mur)
        pt3 = (self.Longueur, self.largueur/2, self.Hauteur)
        pt4  = (0, self.largueur/2, self.Hauteur)
        pente_1 = [[pt1, pt2, pt3,pt4]]
        self.ax.add_collection3d(Poly3DCollection(pente_1,color='r',alpha= 0.5))
        
        pt5 = (self.Longueur , self.largueur + self.debord , self.h_mur)
        pt6 = (0, self.largueur + self.debord, self.h_mur)
        pente_2 = [[pt3, pt5, pt6, pt4]]
        self.ax.add_collection3d(Poly3DCollection(pente_2,color='r',alpha= 0.5))
        return
    
    def plot(self):
        self.ax.set(xlim=(-1, self.Longueur + 1), ylim=(-1, self.largueur + 1), zlim=(-1, self.Hauteur + 1),
               xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        plt.show()
    
    def toiture_4pents(self) :
        """ plot une toiture 4 pents à partir de largeur, longueur et hauteur de la toiture.
        """ 
        
        d = 2
        pt1 = (-self.debord, -self.debord, self.h_mur)
        pt2 = (self.Longueur + self.debord , -self.debord, self.h_mur)
        pt3 = (self.Longueur + self.debord - d, self.largueur/2, self.Hauteur)
        pt4  = (-self.debord + d, self.largueur/2, self.Hauteur)
        pente_1 = [[pt1, pt2, pt3,pt4]]
        self.ax.add_collection3d(Poly3DCollection(pente_1,color='r',alpha= 0.5))
        
        pt5 = pt3
        pt6 = (self.Longueur + self.debord ,self.largueur + self.debord,self.h_mur)
        pt7 = (-self.debord,self.largueur + self.debord,self.h_mur)
        pt8  = pt4
        pente_2 = [[pt5, pt6, pt7, pt8]]
        self.ax.add_collection3d(Poly3DCollection(pente_2,color='r',alpha= 0.5))
        
        pente_3 = [[pt3, pt2, pt6]]
        self.ax.add_collection3d(Poly3DCollection(pente_3,color='r',alpha= 0.5))
        
        pente_4 = [[pt1, pt4, pt7]]
        self.ax.add_collection3d(Poly3DCollection(pente_4,color='r',alpha= 0.5))
        return
    
def toiture() :
    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    x = [0, 1, 1]
    y = [0, 0, 1]
    z = [0, 1, 0]
    
    pt1 = (0,0,0)
    pt2 = (1,0,0)
    pt3 = (1,1,1)
    pt4  = (0,1,1)
    
    verts = [list(zip(x, y, z))]
    verts = [[pt1, pt2, pt3,pt4]]
    print(verts)
    #poly = PolyCollection(verts, facecolors='red', alpha=.7)
    #ax.add_collection3d(poly, zs=lambdas, zdir='y')
    ax.add_collection3d(Poly3DCollection(verts,color='r',alpha= 0.5))
    
    pt5 = (1,1,1)
    pt6 = (1,2,0)
    pt7 = (0,2,0)
    pt8  = (0,1,1)
    verts = [[pt5, pt6, pt7,pt8]]
    ax.add_collection3d(Poly3DCollection(verts,color='r',alpha= 0.5))
    
    ax.set(xlim=(-2, 2), ylim=(-1, 2), zlim=(-1, 2),
           xlabel='x', ylabel=r'$\lambda$', zlabel='probability')
    
    for angle in range(0, 360, 10):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.0001)
        
if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    H,h,L,l,d = 8, 3, 10, 8, 0.6
    
    m = maison(ax,H,h,L,l,d)
    m.maconnerie()
    m.toiture(type = '2_pentes')
    m.plot()

#plt.show()