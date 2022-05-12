# -*- coding: utf-8 -*-
"""
Created on Thu May 12 00:13:48 2022

@author: ngameiro

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

def poutre(ax, pt1, pt2, e, c,  angle = 0) : 
    #L = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    dy = np.cos(angle)*e
    dx = np.sin(angle)*e
    x = [pt2[0] ,
         pt2[0] ,
         pt1[0] ,
         pt1[0] ,
         pt2[0]]
    y = [pt2[1] , 
        pt2[1] - dy ,
        pt1[1] ,
        pt1[1] + dy , 
        pt2[1]]
    ax.add_patch(Polygon(xy=list(zip(x,y)), fill=True, color=c,alpha = 0.5,lw=0))
    return 

def bois(ax , pt1 , pt2 , e , c,  loc) : 
    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]
    dx, dy = x2 - x1, y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    a = np.arctan(dy/dx)
    ey = e/np.cos(a)
    x = [x1, x2 , x2 , x1 , x1]
    if loc == "sup" : 
        y = [y1 - ey , y2 - ey , y2 , y1 , y1 - ey]
    elif loc == "inf" : 
        y = [y1 , y2 , y2 + ey , y1 + ey , y1 ]
    ax.add_patch(Polygon(xy=list(zip(x,y)), fill=True, color=c,alpha = 0.5,lw=0))
    return print("angle = ", np.rad2deg(a) )


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

#########################################
##### PARAMETRAGE #######################
#########################################

#0. type de charpente1 : 
# choix : symetrique / autoporteuse / panne_sur_pignon / tradi_avec_ferme
type_charpente1 = "symetrique"

#1. affiche le cote rue qui est gauche du point 0.0 

#2. epaisseur dalle beton arme
ep_dalle = 20 # cm

#3. epaisseur du batiment
largeur_batiment = 800 #cm portée exterieur mur
axe_ferme = 400

#4. epaisseur de mur
ep_mur1 = 20 # cm
ep_mur2 = 20 #cm

#5. hauteur de mur
h_mur1 = 300 # cm
h_mur2 = 300 # cm

#definir une position a part sinon par defaut au niveau du mur le plus bas

h_archi = 600 #cm par rapport au haut de la dalle (et en bas de la dalle !)
ep_couv = 10 #cm
ep_chevron = 10 #cm

debord_gauche = 40 #cm
debord_droite = 40 #cm

# Dim sabliere
offset_sablier = 4 #cm
h_sablier = 16 #cm
b_sablier = 16 #cm

# Dim chevron
ep_chevron = 10 #cm

#Dim ferme
ep_panne = 26 #cm
b_panne = 10
h_panne = 26
ep_arba = 26 #cm
h_entrait = 20 #cm
b_poinçon = 20 #cm

b_rive = 2.5 #cm
r_gout = 6

pente1 = np.arctan((h_archi-h_mur1)/(axe_ferme+debord_gauche)) #degre
pente2 = np.arctan((h_archi-h_mur2)/(largeur_batiment-axe_ferme+debord_droite)) #degre
a1 = np.round(np.rad2deg(pente1),2)
a2 = np.round(np.rad2deg(pente2),2)
print("pente à gauche : ", a1," deg")
print("pente à droite : ", a2," deg")

### Ajout de la maçonnerie
def walls(ax , ep_dalle , largeur_batiment, ep_mur1, h_mur1 , typ , ep_mur2 = 20 , h_mur2 =300 ) :
    ax.add_patch(Rectangle((0, 0), largeur_batiment, ep_dalle, color='r', fill=False,lw=0.3, hatch='//',label="dalle béton"))
    ax.add_patch(Rectangle((0, ep_dalle), ep_mur1, h_mur1, fill=False,lw=0.3, hatch='\\\\', label="mur"))
    if typ == "sym" : 
        ax.add_patch(Rectangle((largeur_batiment-ep_mur1, ep_dalle), ep_mur1, h_mur1, fill=False,lw=0.3, hatch='\\\\'))
    elif typ == "nosym" : 
        ax.add_patch(Rectangle((largeur_batiment-ep_mur2, ep_dalle), ep_mur2, h_mur2, fill=False,lw=0.3, hatch='\\\\'))
    return print("murs créés")

def echantignolle(ax ,pt , h , r = 0 , reverse = False) : 
    x2 , x3 = pt[0] - 1.5*h , pt[0]
    y2 , y3 = pt[1] , pt[1] + h
    x2 , y2 = rotate((pt[0] , pt[1]) ,(x2 , y2), r)
    x3 , y3 = rotate(pt , [x3 , y3], r)
    x_echan = [pt[0] , x2 , x3]
    y_echan = [pt[1] , y2 , y3]
    if reverse == True : 
        x2 , x3 = pt[0] , pt[0] +  h
        y2 , y3 = pt[1] - 1.5*h , pt[1]
        x2 , y2 = rotate((pt[0] , pt[1]) ,(x2 , y2), r)
        x3 , y3 = rotate(pt , [x3 , y3], r)
        x_echan = [pt[0] , x2 , x3]
        y_echan = [pt[1] , y2 , y3]
    ax1.add_patch(Polygon(xy=list(zip(x_echan,y_echan)), fill=True, color='k',lw=0,label = "echantignolle"))
    return print("echantignolle ajoutée ! ")


def charge(ax,pt1,pt2, q) :
    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]
    dx, dy = x2 - x1, y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    a = np.arctan(dy/dx)
    x = np.linspace(pt1[0], pt2[0],21)
    y = np.linspace(pt1[1], pt2[1],21)
    for i in range(0,21) :
        ax.arrow(x[i],  # x1
                  y[i]+100,  # y1
                  0, # x2 - x1
                  -100, # y2 - y1
                  color='r',
                  lw = 0.3,
                  length_includes_head=True,
                  head_width=10,
                  head_length=10)
        ax.plot([pt1[0],pt2[0]],[pt1[1]+100,pt2[1]+100],lw=0.3,color='r')
        ax.annotate("q = " + str(q) + " kN/m", xy=(pt1[0], pt1[1]), xytext=(abs(pt2[0] - pt1[0])/2, pt1[1] + abs(pt2[1] - pt1[1])/2 + 150 ),fontsize=5)
    return

#poutre(ax1, pt1 = [0,0], pt2 = [100,200], e = 20, c = 'r' , angle = pente1 )
#bois(ax1, pt1 = [0,0], pt2 = [100,200], e = 20, c = 'r' , loc ="sup")

def charpente(ax1 ,axe_ferme , h_archi , offset_sablier , b_sablier , h_sablier , 
              debord_gauche , debord_droite , ep_couv , ep_chevron , 
              ep_panne, b_panne, h_panne , 
              ep_arba , 
              h_entrait , 
              b_poinçon ) :  
    ### Axe ferme
    x_axe = [axe_ferme, axe_ferme]
    y_axe = [0, 1.3*h_archi]
    ax1.plot(x_axe,y_axe,linestyle='dashed',lw=0.4,label="axe ferme")
    
    ### Ajout de la sabliere
    ax1.add_patch(Rectangle((offset_sablier, h_mur1+ep_dalle), b_sablier, h_sablier, color='m', fill=True,lw=0,label='sabliere'))
    ax1.add_patch(Rectangle((largeur_batiment - offset_sablier - b_sablier , h_mur2+ep_dalle), b_sablier, h_sablier, color='m', fill=True,lw=0))
    
    ### Ajout des chevrons
    #gauche
    delta_y1 = np.tan(pente1)*(debord_gauche + offset_sablier)
    pt1_chev1 = [-debord_gauche , h_mur1+ep_dalle+h_sablier-delta_y1 ]
    pt2_chev1 = [axe_ferme , h_archi + ep_dalle - ep_couv/np.cos(pente1) - ep_chevron/np.cos(pente1)]
    bois(ax1, pt1_chev1, pt2_chev1, e = ep_chevron , c = "g", loc = "inf")
    # droite
    delta_y2 = np.tan(pente2)*(debord_droite + offset_sablier)
    pt1_chev2 = [axe_ferme , h_archi + ep_dalle - ep_couv/np.cos(pente1) -ep_chevron/np.cos(pente1)]
    pt2_chev2 = [largeur_batiment+debord_droite , h_mur2 + ep_dalle + h_sablier - delta_y2]
    bois(ax1, pt1_chev2, pt2_chev2, e = ep_chevron , c = "g", loc = "inf")
    
    ### Ajout de la couverture
    #gauche
    pt1_couv1 = [pt1_chev1[0] , pt1_chev1[1] + ep_chevron/np.cos(pente1) ]
    pt2_couv1 = [pt2_chev1[0] , pt2_chev1[1] + ep_chevron/np.cos(pente1) ]
    bois(ax1, pt1_couv1, pt2_couv1, e = ep_couv , c = "r", loc = "inf")
    ax1.text(0.25*axe_ferme ,h_archi*1.0 , "pente1 = "+str(a1)+"°",fontsize=8 ,color='r')

    # droite
    pt1_couv2 = [pt1_chev2[0] , pt1_chev2[1] + ep_chevron/np.cos(pente1) ]
    pt2_couv2 = [pt2_chev2[0] , pt2_chev2[1] + ep_chevron/np.cos(pente1) ]
    bois(ax1, pt1_couv2, pt2_couv2, e = ep_couv , c = "r", loc = "inf")
    ax1.text(0.75*largeur_batiment ,h_archi*1.0, "pente2 = "+str(a2)+"°", fontsize=8,color='r')

    
    ### Ajout de chambre de panne
    #gauche
    dy = np.tan(pente1)*b_sablier
    pt1_panne1 = [offset_sablier + b_sablier , h_mur1 + ep_dalle + h_sablier + dy]
    pt2_panne1 = pt2_chev1
    bois(ax1, pt1_panne1, pt2_panne1, e = ep_panne , c = "gray", loc = "sup")
    
    #droite
    dy = np.tan(pente2)*b_sablier
    pt1_panne2 = pt1_chev2
    pt2_panne2 = [ largeur_batiment - offset_sablier - b_sablier , h_mur2 + ep_dalle + h_sablier + dy]
    bois(ax1, pt1_panne2, pt2_panne2, e = ep_panne , c = "gray", loc = "sup")
    
    ey = ep_panne/np.cos(pente1)
    Type_panne = 'devers' #ou 'aplomb'
    if Type_panne == 'devers':
        #gauche
        L = np.sqrt((pt2_panne1[0] - pt1_panne1[0])**2 + (pt2_panne1[1] - pt1_panne1[1])**2)
        print(L,L//180)
        for i in range(int(L // 180)) :      
            x_panne = pt1_panne1[0] + np.cos(pente1)*L/(L // 180 + 1)*(i+1)
            y_panne = pt1_panne1[1] - ey + np.sin(pente1)*L/(L // 180 + 1)*(i+1)
            ax1.add_patch(Rectangle((x_panne , y_panne), b_panne, h_panne, color='gray', fill=True, lw=0, angle= a1))
            pt1 = [x_panne,y_panne]
            echantignolle(ax1 , pt1 , h_panne , pente1 )
        #droite
        L = np.sqrt((pt2_panne2[0] - pt1_panne2[0])**2 + (pt2_panne2[1] - pt1_panne2[1])**2)
        print(L,L//180)
        for i in range(int(L // 180)) :      
            x_panne = pt2_panne2[0] - np.cos(pente2)*L/(L // 180 + 1)*(i+1)
            y_panne = pt2_panne2[1] - ey + np.sin(pente2)*L/(L // 180 + 1)*(i+1)
            ax1.add_patch(Rectangle((x_panne , y_panne), -b_panne, h_panne, color='gray', fill=True, lw=0 , angle= -a2))
            pt1 = [x_panne,y_panne]
            echantignolle(ax1 , pt1 , h_panne, np.pi/2-pente2 , True )
    elif Type_panne == 'aplomb' :
        x_panne = [largeur_batiment/2, largeur_batiment/2, 0 , 0 , largeur_batiment/2]
        y_panne = [h_archi+ep_dalle-ep_couv-ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne, h_mur1+ep_dalle+ep_couv+ep_chevron-ep_panne, h_mur1+ep_dalle+ep_couv+ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron]
        

    
    #### attention ####
    # les ep indiquées ne sont pas les bonnes !
    ###################

    ### Ajout de chambre d'arba
    #gauche
    pt1_arba1 = [pt1_panne1[0] , pt1_panne1[1] - ep_panne/np.cos(pente1) - ep_arba/np.cos(pente1) ]
    pt2_arba1 = [pt2_panne1[0] , pt2_panne1[1] - ep_panne/np.cos(pente1) - ep_arba/np.cos(pente1) ]
    bois(ax1, pt1_arba1, pt2_arba1, e = ep_arba , c = "y", loc = "inf")

    # droite
    pt1_arba2 = [pt1_panne2[0] , pt1_panne2[1] - ep_panne/np.cos(pente1) - ep_arba/np.cos(pente1) ]
    pt2_arba2 = [pt2_panne2[0] , pt2_panne2[1] - ep_panne/np.cos(pente1) - ep_arba/np.cos(pente1) ]
    bois(ax1, pt1_arba2, pt2_arba2, e = ep_arba , c = "y", loc = "inf")
    
    # Entrait
    L_entrait = largeur_batiment-ep_mur1-ep_mur2
    ax1.add_patch(Rectangle((ep_mur1, min(h_mur1,h_mur2)), L_entrait, h_entrait , color='k', fill=False, lw=0.5,label="entrait"))
    ### Ferme
    L_poinçon = h_archi-h_mur1-ep_couv+h_entrait #cm
    ax1.add_patch(Rectangle((axe_ferme-b_poinçon/2, h_mur1+ep_dalle-h_entrait), b_poinçon, L_poinçon , color='b', alpha=0.5, fill=True, lw=0,label="poinçon"))
    
    """
    #jambe de force
    h_jdf = 0.5*L_poinçon + h_mur1 + ep_dalle
    b_jdf = 20 #cm
    a_jdf = np.deg2rad(45)
    x_jdf = [axe_ferme-b_poinçon/2 ,
             axe_ferme-b_poinçon/2 ,
             axe_ferme-b_poinçon/2 - np.tan(a_jdf)*100 ,
             axe_ferme-b_poinçon/2 - np.tan(a_jdf)*(100+b_jdf) , 
             axe_ferme-b_poinçon/2]
    y_jdf = [h_jdf, 
             h_jdf-b_jdf ,
             h_jdf+10 , 
             h_jdf+10+b_jdf ,
             h_jdf]
    ax1.add_patch(Polygon(xy=list(zip(x_jdf,y_jdf)), fill=True, color='m',alpha = 1,lw=0,label='jdf'))
    """
    
    return print("charpente créée")

def couverture() :
    ### Planche de rive
    h_rive = ep_couv + ep_chevron + 2 #cm
    delta_y1 = np.tan(pente1)*(debord_gauche + offset_sablier)
    ax1.add_patch(Rectangle((-debord_gauche -b_rive,h_mur1 + ep_dalle +h_sablier-delta_y1 - 2, 0), b_rive, h_rive, color='gray', fill=True, lw=0,label="rive"))

    ### Goutiere
    filled_arc((-debord_gauche-r_gout - b_rive ,h_mur1 + ep_dalle +h_sablier-delta_y1), r_gout, 180, 360, ax1, "k","goutiere")
    return print("couverture crée")


if __name__ == "__main__" : 
    # plot
    plt.figure(figsize=(11.69,8.27)) # for landscape
    fig, ax1 = plt.subplots()
    
    #construction
    walls(ax1 , ep_dalle , largeur_batiment, ep_mur1, h_mur1 , "sym" )
    
    charpente(ax1 ,axe_ferme , h_archi , offset_sablier , b_sablier , h_sablier , 
                  debord_gauche , debord_droite , ep_couv , ep_chevron , 
                  ep_panne, b_panne, h_panne , 
                  ep_arba , 
                  h_entrait , 
                  b_poinçon )
    
    couverture()

    #charge(ax1,pt1_couv1, pt2_couv1, 10)
    #charge(ax1,pt1_couv2, pt2_couv2, 10)
    #charge(ax1,[ep_mur1,h_mur1+h_entrait], [ep_mur1 + L_entrait,h_mur1+h_entrait])

    ax1.axis('equal')
    #plt.grid(color = 'b', linestyle = '--', linewidth = 0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
    #ax1.axis('off')
    
    fig.savefig('schema_ferme.png', format='png', dpi=200)
