# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:44:52 2022

@author: ngameiro
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:38:57 2022

@author: ngameiro

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
largeur_batiment = 600 #cm portée exterieur mur
axe_ferme = 300

#4. epaisseur de mur
ep_mur1 = 20 # cm
ep_mur2 = 20 #cm

#5. hauteur de mur
h_mur1 = 210 # cm
h_mur2 = 210 # cm

#definir une position a part sinon par defaut au niveau du mur le plus bas

h_archi = 380 #cm par rapport au haut de la dalle (et en bas de la dalle !)
ep_couv = 10 #cm
ep_chevron = 10 #cm

debord_gauche = 40 #cm
debord_droite = 40 #cm

# Dim sabliere
offset_sablier = 6 #cm
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

# plot
plt.figure(figsize=(11.69,8.27)) # for landscape
fig, ax1 = plt.subplots()

### Ajout de la maçonnerie
ax1.add_patch(Rectangle((0, 0), largeur_batiment, ep_dalle, color='r', fill=False,lw=0.3, hatch='//',label="dalle béton"))
ax1.add_patch(Rectangle((0, ep_dalle), ep_mur1, h_mur1, fill=False,lw=0.3, hatch='\\\\', label="mur"))
ax1.add_patch(Rectangle((largeur_batiment-ep_mur2, ep_dalle), ep_mur2, h_mur2, fill=False,lw=0.3, hatch='\\\\'))

### Axe ferme
x_axe = [axe_ferme, axe_ferme]
y_axe = [0, 1.3*h_archi]
ax1.plot(x_axe,y_axe,linestyle='dashed',lw=0.4,label="axe ferme")

### Ajout de la sabliere
ax1.add_patch(Rectangle((offset_sablier, h_mur1+ep_dalle), b_sablier, h_sablier, color='m', fill=True,lw=0,label='sabliere'))
ax1.add_patch(Rectangle((largeur_batiment - offset_sablier - b_sablier , h_mur2+ep_dalle), b_sablier, h_sablier, color='m', fill=True,lw=0))

### Ajout des chevrons
# gauche
x_chevron1 = [axe_ferme, axe_ferme, offset_sablier , offset_sablier , axe_ferme]
y_chevron1 = [h_archi+ep_dalle-ep_couv , 
              h_archi+ep_dalle-ep_couv-ep_chevron , 
              h_mur1+ep_dalle+h_sablier , 
              h_mur1+ep_dalle+ep_chevron+h_sablier , 
              h_archi+ep_dalle-ep_couv]
ax1.add_patch(Polygon(xy=list(zip(x_chevron1,y_chevron1)), fill=True, color='g',alpha = 0.5,lw=0,label='chevron'))
#07/05/2022
x_debord = [offset_sablier, offset_sablier, -debord_gauche, -debord_gauche , offset_sablier]
delta_y1 = np.tan(pente1)*debord_gauche
y_debord1 = [h_mur1+ep_dalle+h_sablier+ep_chevron , 
             h_mur1+ep_dalle+h_sablier , 
             h_mur1+ep_dalle+h_sablier-delta_y1,
             h_mur1+ep_dalle+h_sablier-delta_y1 + ep_chevron,
             h_mur1+ep_dalle+h_sablier+ep_chevron]
ax1.add_patch(Polygon(xy=list(zip(x_debord,y_debord1)), fill=True, color='g',alpha = 0.5,lw=0))
# droite
delta_y2 = np.tan(pente2)*debord_droite
x_chevron2 = [axe_ferme, axe_ferme, largeur_batiment+debord_droite , largeur_batiment+debord_droite , axe_ferme]
y_chevron2 = [h_archi + ep_dalle - ep_couv , 
              h_archi + ep_dalle - ep_couv - ep_chevron ,
              h_mur2 + ep_dalle + h_sablier - delta_y2 ,
              h_mur2 + ep_dalle + h_sablier - delta_y2 + ep_chevron , 
              h_archi + ep_dalle - ep_couv]
ax1.add_patch(Polygon(xy=list(zip(x_chevron2,y_chevron2)), fill=True, color='g',alpha = 0.5,lw=0))

### Ajout de la couverture
# gauche
x_couv1 = [axe_ferme, axe_ferme, offset_sablier , offset_sablier , axe_ferme]
y_couv1 = [h_archi+ep_dalle, h_archi+ep_dalle-ep_couv, h_mur1+ep_dalle+h_sablier+ep_chevron , h_mur1+ep_dalle+h_sablier+ep_chevron+ep_couv , h_archi+ep_dalle]
ax1.add_patch(Polygon(xy=list(zip(x_couv1,y_couv1)), fill=True, color='r',alpha = 0.5,lw=0,label='couverture'))
ax1.text(0.25*axe_ferme ,h_archi*1.0 , "pente1 = "+str(a1)+"°",fontsize=8 ,color='r')

x_debord = [offset_sablier, offset_sablier, -debord_gauche, -debord_gauche , offset_sablier]
y_debord2 = [h_mur1 + ep_dalle + h_sablier + ep_chevron + ep_couv ,
             h_mur1+ep_dalle+h_sablier+ep_chevron ,
             h_mur1+ep_dalle+h_sablier-delta_y1 + ep_chevron ,
             h_mur1+ep_dalle+h_sablier-delta_y1 + ep_chevron+ep_couv ,
             h_mur1+ep_dalle+h_sablier+ep_chevron+ep_couv ]
ax1.add_patch(Polygon(xy=list(zip(x_debord,y_debord2)), fill=True, color='r',alpha = 0.5,lw=0,label='couverture'))

# droite
x_couv2 = [axe_ferme, axe_ferme, largeur_batiment+debord_droite , largeur_batiment+debord_droite , axe_ferme]
y_couv2 = [h_archi + ep_dalle ,
           h_archi + ep_dalle - ep_couv ,
           h_mur2 + ep_dalle + h_sablier - delta_y2 + ep_chevron,
           h_mur2 + ep_dalle + h_sablier - delta_y2 + ep_chevron + ep_couv ,
           h_archi + ep_dalle]
ax1.add_patch(Polygon(xy=list(zip(x_couv2,y_couv2)), fill=True, color='r',alpha = 0.5,lw=0))
ax1.text(0.75*largeur_batiment ,h_archi*1.0, "pente2 = "+str(a2)+"°", fontsize=8,color='r')


### Ajout de chambre de panne
Type_panne = 'devers' #ou 'aplomb'
if Type_panne == 'devers':
    x_panne1 = [axe_ferme, axe_ferme, ep_mur1 , ep_mur1, axe_ferme]
    y_panne1 = [h_archi+ep_dalle-ep_couv-ep_chevron,
                h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne, 
                h_archi - (axe_ferme - ep_mur1)*np.tan(pente1) - ep_panne, 
                h_archi - (axe_ferme - ep_mur1)*np.tan(pente1) , 
                h_archi+ep_dalle-ep_couv-ep_chevron]
    L = np.sqrt((axe_ferme - ep_mur1)**2 + (h_archi - ep_couv - ep_chevron - h_mur1)**2)
    print(L,L//180)
    for i in range(int(L // 180)) :      
        x_panne = ep_mur1 + np.cos(pente1)*L/(L // 180 + 1)*(i+1)
        y_panne = h_mur1 + np.sin(pente1)*L/(L // 180 + 1)*(i+1)
        ax1.add_patch(Rectangle((x_panne , y_panne), b_panne, h_panne, color='gray', fill=True, lw=0,label="panne", angle=a1))
        x_echan = [x_panne, x_panne, x_panne - 1.5*h_panne, x_panne]
        y_echan = [y_panne + h_panne , y_panne , y_panne , y_panne + h_panne]
        ax1.add_patch(Polygon(xy=list(zip(x_echan,y_echan)), fill=True, color='k',lw=0,label = "echantignolle"))
elif Type_panne == 'aplomb' :
    x_panne = [largeur_batiment/2, largeur_batiment/2, 0 , 0 , largeur_batiment/2]
    y_panne = [h_archi+ep_dalle-ep_couv-ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne, h_mur1+ep_dalle+ep_couv+ep_chevron-ep_panne, h_mur1+ep_dalle+ep_couv+ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron]
    
ax1.add_patch(Polygon(xy=list(zip(x_panne1,y_panne1)), fill=True, color='k',alpha = 0.1,lw=0,label='panne'))

### Ajout de chambre d'arba
x_arba = [axe_ferme, axe_ferme, ep_mur1 , ep_mur1, axe_ferme]
y_arba = [h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne ,
          h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne-ep_arba,
          h_mur1+ep_dalle+ep_couv+ep_chevron-ep_panne-ep_arba,
          h_mur1+ep_dalle+ep_couv+ep_chevron-ep_panne,
          h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne]
ax1.add_patch(Polygon(xy=list(zip(x_arba,y_arba)), fill=True, color='y',alpha = 0.5,lw=0,label='arba'))

### Planche de rive
h_rive = ep_couv+ep_chevron #cm
ax1.add_patch(Rectangle((-debord_gauche-b_rive,h_mur1+ep_couv+ep_chevron, 0), b_rive, h_rive, color='gray', fill=True, lw=0,label="rive"))

### Goutiere
filled_arc((-debord_gauche-r_gout,h_mur1+ep_couv+ep_chevron), r_gout, 180, 360, ax1, "k","goutiere")

# Entrait
L_entrait = largeur_batiment-ep_mur1-ep_mur2
ax1.add_patch(Rectangle((ep_mur1, min(h_mur1,h_mur2)), L_entrait, h_entrait , color='k', fill=False, lw=0.5,label="entrait"))
### Ferme
L_poinçon = h_archi-h_mur1-ep_couv+h_entrait #cm
ax1.add_patch(Rectangle((axe_ferme-b_poinçon/2, h_mur1+ep_dalle-h_entrait), b_poinçon, L_poinçon , color='b', alpha=0.5, fill=True, lw=0,label="poinçon"))
# Entrait

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

def charge_horizontale(ax,x,y,L) :
    for i in range(0,21) :
        ax1.arrow(x + i/20*L,  # x1
                  y+100,  # y1
                  0, # x2 - x1
                  -100, # y2 - y1
                  color='r',
                  lw = 0.3,
                  length_includes_head=True,
                  head_width=10,
                  head_length=10)
        ax1.plot([x,x+L],[y+100,y+100],lw=0.3,color='r')
        ax1.annotate("q = 5 kN/m", xy=(0, 0), xytext=(x+L/2, y + 150),fontsize=5)
    return

#poutre(ax1, pt1 = [0,0], pt2 = [100,200], e = 20, c = 'r' , angle = pente1 )
bois(ax1, pt1 = [0,0], pt2 = [100,200], e = 20, c = 'r' , loc ="sup")

# charge_horizontale(ax1,ep_mur1,h_mur1+h_entrait,L_entrait)

ax1.axis('equal')
#plt.grid(color = 'b', linestyle = '--', linewidth = 0.5)
ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
#ax1.axis('off')

plt.show()
fig.savefig('schema_ferme.png', format='png', dpi=200)
