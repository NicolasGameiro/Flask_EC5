# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:17:35 2022

@author: ngameiro
"""

import Code_FEM_v5 as fem
import tracer_charpente_v5 as tc5
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def projet():
    p = 6.5 + 0.8*2
    h = 1.93
    
    #Maillage
    mesh = fem.Mesh(2)
    mesh.add_node([0,0])
    mesh.add_node([p/2,0])
    mesh.add_node([p,0])
    mesh.add_node([p/2,h])
    mesh.add_element([1,2], "entrait", "r", 22 ,10, 10)
    mesh.add_element([2,3], "entrait", "r", 22 ,10, 10)
    mesh.add_element([3,4], "arba", "b", 22 ,10)
    mesh.add_element([4,2], "poinçon", "m", 15 ,15)
    mesh.add_element([4,1], "arba", "b", 22 ,10)
    mesh.plot_mesh()
    
    #Modele
    f = fem.FEM_Model(mesh)
    f.mesh.plot_mesh()
    # bc
    f.apply_bc([1,1,0],1)
    f.apply_bc([1,1,0],21)
    f.apply_distributed_load(1000, [1,21])
    f.plot_forces(type = 'dist', pic = False)
    f.solver_frame()
    U, React, res = f.get_res()
    f.U_table()
    f.R_table()
    f.plot_disp_f(dir='y', pic = True)
    #f.rapport()
    f.stress()
    f.S_table()
    f.plot_stress(s="sf")
    

if __name__ == "__main__":
    projet()