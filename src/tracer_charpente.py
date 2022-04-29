# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:38:57 2022

@author: ngameiro
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12


# plot
fig, ax = plt.subplots()
ep_dalle = 20 # cm
portee = 800 #cm portée exterieur mur
ep_mur = 20 # cm
h_mur = 300 # cm
h_archi = 700 #cm
ep_couv = 10 #cm
ep_chevron = 10 #cm
pente = 40 #degre
a = np.deg2rad(pente)
sablier = (h_archi-h_mur)/np.tan(a)
x = [0, 0, portee, portee, ep_mur, 0, ep_mur, portee/2, portee/2,portee/2-sablier,portee/2-sablier]
y = [0, ep_dalle, 0, ep_dalle, ep_dalle, h_mur, h_mur, h_archi, h_archi-ep_couv,h_mur,h_mur-ep_couv]

ax.scatter(x, y , s=1,color='r')
ax.annotate("",xy=(portee,-20), xycoords='data',xytext=(0,-20), textcoords='data',arrowprops=dict(arrowstyle="<->",connectionstyle="arc3",color='r'), )
ax.text(portee/3,-50, "portée", fontsize=8,color='r')
x1 = [0,portee,portee,ep_mur,ep_mur,0,0,0]
y1 = [0,0,ep_dalle,ep_dalle,h_mur,h_mur,ep_dalle,0]
ax.plot(x1,y1)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

#plt.xlim(-50, max(x)+100)
#plt.xlabel("x", fontsize=12)            
#plt.ylim(-50, max(y)+100)                                    
#plt.ylabel("y", fontsize=12)  
#fig.tight_layout()
ax.axis('equal')
plt.show()
