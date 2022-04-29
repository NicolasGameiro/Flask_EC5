# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:28:50 2022

@author: ngameiro
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:38:57 2022

@author: ngameiro

TO DO :
- Entre un coté rue (à gauche 0 ;0)
- Entre un axe de ferme (par rapport a l’extérieur au mur de gauche)
- Entre une section de sablière (entree une fixation de sablière en jaune) 
puis une distance de placement de l’arrete basse gauche de la sablière par rapport a l’exterieur du mur de gauche

FAIT :
- Entree une épaisseur de dalle béaton armée
- Entre une distance de largeur du batiment (exterieur mur)
- epaisseur de mur
- Entre hauteur mur gauche 
- Entree une hauteur maximale par rapport au dessus de la dalle
- Descendre de 10 cm couverture
- Creation epaisseur couverture en rouge transparent
- Decsnedre d’une hauteur a rentree (d’épaisseur chevrons)
- Entre une distance de débord a gauche par rapport a l’exterieur mur
- Création latti de chevrons avec son épaisseur
- Affichage de la pente au dessus du chevrons de gauche
- Affichage planche de rive 
- Affichage gouttiere

V3 :
- Epaisseur mur gauche entre epaisseru mur droite
- mur beton avec representation ceinture beton arme de 12cm vers le bas
- hauteur mur de droite
- sabliere à droite différente
- debord à droite différent
- penteà droite
- Affichage planche de rive détaillé plus tardépaiseur , cheneau, chanlatte…
- Affichage gouttiere detaillé plus tard demi ronde carre rampant…

 
Charpente autoporteuse
Charpente panne sur pignons
Charpente traditionnel avec ferme 
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

#0. type de charpente : 
# choix : symetrique / autoporteuse / panne_sur_pignon / tradi_avec_ferme
type_charpente = "symetrique"

#1. affiche le cote rue qui est gauche du point 0.0 

#2. epaisseur dalle beton arme
ep_dalle = 20 # cm

#3. epaisseur du batiment
largeur_batiment = 800 #cm portée exterieur mur

#4. epaisseur de mur
ep_mur = 20 # cm
ep_mur_2 = 20 #cm

#5. hauteur de mur
h_mur = 300 # cm
h_mur2 = 300 # cm

h_archi = 700 #cm par rapport au haut de la dalle (et en bas de la dalle !)
ep_couv = 10 #cm
ep_chevron = 10 #cm

debord_gauche = 40 #cm
debord_droite = 40 #cm

pente = np.arctan((h_archi-h_mur)/(largeur_batiment/2+debord_gauche)) #degre
a = np.round(np.rad2deg(pente),2)
print("pente à gauche : ", a," deg")

# plot
plt.figure(figsize=(11.69,8.27)) # for landscape
fig, ax1 = plt.subplots()

### Ajout de la maçonnerie
ax1.add_patch(Rectangle((0, 0), largeur_batiment, ep_dalle, color='r', fill=False, hatch='//',label="dalle béton"))
ax1.add_patch(Rectangle((0, ep_dalle), ep_mur, h_mur, fill=False, hatch='\\\\', label="mur"))
ax1.add_patch(Rectangle((largeur_batiment-ep_mur, ep_dalle), ep_mur, h_mur, fill=False, hatch='\\\\'))

### Ajout de la couverture
x_couv = [largeur_batiment/2, largeur_batiment/2, -debord_gauche , -debord_gauche , largeur_batiment/2]
y_couv = [h_archi+ep_dalle, h_archi+ep_dalle-ep_couv, h_mur-ep_couv, h_mur, h_archi+ep_dalle]
ax1.add_patch(Polygon(xy=list(zip(x_couv,y_couv)), fill=True, color='r',alpha = 0.5,lw=0,label='couverture'))
ax1.text(0.1*largeur_batiment ,h_archi*0.7, "pente = "+str(a)+"°", fontsize=8,color='r',rotation=a)

### Ajout des chevrons
ep_chevron = 10 #cm
x_chevron = [largeur_batiment/2, largeur_batiment/2, -debord_gauche , -debord_gauche , largeur_batiment/2]
y_chevron = [h_archi+ep_dalle-ep_couv, h_archi+ep_dalle-ep_couv-ep_chevron, h_mur-ep_couv-ep_chevron, h_mur-ep_couv, h_archi+ep_dalle-ep_couv]
ax1.add_patch(Polygon(xy=list(zip(x_chevron,y_chevron)), fill=True, color='g',alpha = 0.5,lw=0,label='chevron'))

### Ajout de la sabliere
offset_sablier = 10 #cm
h_sablier = 8 #cm
b_sablier = 8 #cm
ax1.add_patch(Rectangle((offset_sablier, h_mur+ep_dalle), b_sablier, h_sablier, color='m', fill=True,lw=0,label='sabliere'))

### Ajout de chambre de panne
Type_panne = 'devers' #ou 'aplomb'
ep_panne = 26 #cm
if Type_panne == 'devers':
    x_panne = [largeur_batiment/2, largeur_batiment/2, ep_mur , ep_mur , largeur_batiment/2]
    y_panne = [h_archi+ep_dalle-ep_couv-ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne, h_mur+ep_dalle+ep_couv+ep_chevron-ep_panne, h_mur+ep_dalle+ep_couv+ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron]
elif Type_panne == 'aplomb' :
    x_panne = [largeur_batiment/2, largeur_batiment/2, 0 , 0 , largeur_batiment/2]
    y_panne = [h_archi+ep_dalle-ep_couv-ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron-ep_panne, h_mur+ep_dalle+ep_couv+ep_chevron-ep_panne, h_mur+ep_dalle+ep_couv+ep_chevron, h_archi+ep_dalle-ep_couv-ep_chevron]
    
ax1.add_patch(Polygon(xy=list(zip(x_panne,y_panne)), fill=True, color='c',alpha = 0.5,lw=0,label='panne'))

### Planche de rive
h_rive = ep_couv+ep_chevron #cm
b_rive = 5 #cm
ax1.add_patch(Rectangle((-debord_gauche-b_rive,h_mur-ep_couv-ep_chevron, 0), b_rive, h_rive, color='gray', fill=True, lw=0,label="rive"))

### Goutiere
r_gout = 10
filled_arc((-debord_gauche-r_gout,h_mur-ep_couv-ep_chevron), r_gout, 180, 360, ax1, "k","goutiere")

### Ferme
b_poinçon = 20 #cm
L_poinçon = h_archi-h_mur #cm
ax1.add_patch(Rectangle((largeur_batiment/2-b_poinçon/2, h_archi+ep_dalle-L_poinçon), b_poinçon, L_poinçon , color='b', alpha=0.5, fill=True, lw=0,label="poinçon"))
# Entrait
L_entrait = largeur_batiment-2*ep_mur
h_entrait = 20 #cm
ax1.add_patch(Rectangle((ep_mur, h_mur), L_entrait, h_entrait , color='k', fill=False, lw=0.5,label="entrait"))
#jambe de force
h_jdf = 0.5*L_poinçon + h_mur + ep_dalle
b_jdf = 20 #cm
a_jdf = np.deg2rad(45)
x_jdf = [largeur_batiment/2-b_poinçon/2 ,
         largeur_batiment/2-b_poinçon/2 ,
         largeur_batiment/2-b_poinçon/2 - np.tan(a_jdf)*100 ,
         largeur_batiment/2-b_poinçon/2 - np.tan(a_jdf)*(100+b_jdf) , 
         largeur_batiment/2-b_poinçon/2]
y_jdf = [h_jdf, 
         h_jdf-b_jdf ,
         h_jdf+10 , 
         h_jdf+10+b_jdf ,
         h_jdf]
ax1.add_patch(Polygon(xy=list(zip(x_jdf,y_jdf)), fill=True, color='m',alpha = 1,lw=0,label='jdf'))



ax1.axis('equal')
plt.grid(color = 'b', linestyle = '--', linewidth = 0.5)
ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.)
#ax1.axis('off')

plt.show()
fig.savefig('schema_ferme.png', format='png', dpi=200)
