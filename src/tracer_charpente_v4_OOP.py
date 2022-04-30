# -*- coding: utf-8 -*-
"""
TO DO :
- passer en orienté objet
- generer un rapport PDF de la charpente avec chaque coupe et des info dessus

FAIT :
- Epaisseur mur gauche entre epaisseru mur droite
- mur beton avec representation ceinture beton arme de 12cm vers le bas
- hauteur mur de droite
- sabliere à droite différente
- debord à droite différent
- pente1à droite
- Affichage planche de rive détaillé plus tardépaiseur , cheneau, chanlatte…
- Affichage gouttiere detaillé plus tard demi ronde carre rampant…

 
Charpente1 autoporteuse
Charpente1 panne sur pignons
Charpente1 traditionnel avec ferme 
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Polygon, Arc, Wedge
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def filled_arc(center, radius, theta1, theta2, ax, color,label):

    circ = Wedge(center, radius, theta1, theta2, fill=True, color=color)
    pt1 = (radius * (np.cos(theta1*np.pi/180.)) + center[0],
           radius * (np.sin(theta1*np.pi/180.)) + center[1])
    pt2 = (radius * (np.cos(theta2*np.pi/180.)) + center[0],
           radius * (np.sin(theta2*np.pi/180.)) + center[1])
    pt3 = center
    pol = Polygon([pt1, pt2, pt3], lw=0, color = color, label= label)
    ax.add_patch(circ)
    ax.add_patch(pol)
    return

class Ferme_Designer() : 
    def __init__(self, ferme_type) :
        self.ferme_type = ferme_type
        self.ep_dalle = 20
        self.largeur_batiment = 800
        self.axe_ferme = self.largeur_batiment/2
        self.ep_mur1 = 20 # cm
        self.ep_mur2 = self.ep_mur1 #cm
        self.h_mur1 = 300 # cm
        self.h_mur2 = self.h_mur1 # cm
        self.h_archi = 700 #cm par rapport au haut de la dalle (et en bas de la dalle !)
        self.ep_couv = 10 #cm
        self.ep_chevron = 10 #cm
        self.debord_gauche = 40 #cm
        self.debord_droite = self.debord_gauche #cm
        self.pente1 = np.arctan((self.h_archi-self.h_mur1)/(self.axe_ferme+self.debord_gauche)) #degre
        self.pente2 = np.arctan((self.h_archi-self.h_mur2)/(self.largeur_batiment-self.axe_ferme+self.debord_droite)) #degre
        #for plot
        # plot
        plt.figure(figsize=(11.69,8.27)) # for landscape
        self.fig, self.ax1 = plt.subplots()
    
    def get_pente(self) : 
        a1 = np.round(np.rad2deg(self.pente1),2)
        a2 = np.round(np.rad2deg(self.pente2),2)
        print("pente à gauche : ", a1," deg")
        print("pente à droite : ", a2," deg")
    
    def draw(self) : 
        self.ax1.axis('equal')
        plt.grid(color = 'b', linestyle = '--', linewidth = 0.5)
        self.ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
        plt.show()
        self.fig.savefig('schema_ferme.png', format='png', dpi=200)

    def add_maconnerie(self, ep_dalle, h_mur1, ep_mur1, h_mur2 = 0, ep_mur2 = 0) : 
        if h_mur2 == 0 : 
            h_mur2 = h_mur1
        if ep_mur2 == 0 : 
            ep_mur2 = ep_mur1
        self.ep_dalle = ep_dalle
        self.ep_mur1 = ep_mur1
        self.ep_mur2 = ep_mur2
        self.h_mur1 = h_mur1
        self.h_mur2 = h_mur2
        ### Ajout de la maçonnerie
        self.ax1.add_patch(Rectangle((0, 0), self.largeur_batiment, self.ep_dalle, color='r', fill=False, hatch='//',label="dalle béton"))
        self.ax1.add_patch(Rectangle((0, self.ep_dalle), self.ep_mur1, self.h_mur1, fill=False, hatch='\\\\', label="mur"))
        self.ax1.add_patch(Rectangle((self.largeur_batiment-self.ep_mur2, self.ep_dalle), self.ep_mur2, self.h_mur2, fill=False, hatch='\\\\'))

    def add_axe(self, axe_ferme, h_archi) : 
        self.axe_ferme = axe_ferme
        self.h_archi = h_archi
        ### Axe ferme
        x_axe = [self.axe_ferme, self.axe_ferme]
        y_axe = [0, 1.3*self.h_archi]
        self.ax1.plot(x_axe,y_axe,linestyle='dashed',lw=0.4,label="axe ferme")
        
    def add_sabliere(self, offset_sablier, h , b ) :
        self.offset_sablier = offset_sablier #cm
        self.h_sablier = h #cm
        self.b_sablier = b #cm
        self.ax1.add_patch(Rectangle((self.offset_sablier, self.h_mur1+self.ep_dalle), self.b_sablier, self.h_sablier, color='m', fill=True,lw=0,label='sabliere'))
        self.ax1.add_patch(Rectangle((self.largeur_batiment - self.offset_sablier, self.h_mur2+self.ep_dalle), self.b_sablier, self.h_sablier, color='m', fill=True,lw=0))

    def add_chevron(self, ep_chevron = 10) : 
        ### Ajout des chevrons
        self.ep_chevron = ep_chevron #cm
        # gauche
        x_chevron1 = [self.axe_ferme, self.axe_ferme, self.offset_sablier , self.offset_sablier , axe_ferme]
        y_chevron1 = [h_archi+ep_dalle-ep_couv, h_archi+ep_dalle-ep_couv-ep_chevron, h_mur1+ep_dalle+h_sablier , h_mur1+ep_dalle+ep_chevron+h_sablier, h_archi+ep_dalle-ep_couv]
        ax1.add_patch(Polygon(xy=list(zip(x_chevron1,y_chevron1)), fill=True, color='g',alpha = 0.5,lw=0,label='chevron'))
        # droite
        x_chevron2 = [axe_ferme, axe_ferme, largeur_batiment+debord_droite , largeur_batiment+debord_droite , axe_ferme]
        y_chevron2 = [h_archi+ep_dalle-ep_couv, h_archi+ep_dalle-ep_couv-ep_chevron, h_mur2-ep_couv-ep_chevron, h_mur2-ep_couv, h_archi+ep_dalle-ep_couv]
        self.ax1.add_patch(Polygon(xy=list(zip(x_chevron2,y_chevron2)), fill=True, color='g',alpha = 0.5,lw=0))

if __name__ == "__main__" :
    f = Ferme_Designer("sym")
    f.add_maconnerie(20,200,20)
    f.add_axe(400,700)
    f.add_sabliere(10,16,16)
    f.draw()

