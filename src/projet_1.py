# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:17:35 2022

@author: ngameiro
"""

import Code_FEM_v5 as fem
import tracer_charpente_v5 as tc5

p = 6.5 + 0.8*2
h = 1.93

#Maillage
mesh = fem.Mesh(2)
mesh.add_node([0,0])
mesh.add_node([p/2,0])
mesh.add_node([p,0])
mesh.add_node([p/2,h])
mesh.add_element([1,2], "entrait", "r", 22 ,10)
mesh.add_element([2,3], "entrait", "r", 22 ,10)
mesh.add_element([3,4], "arba", "b", 22 ,10)
mesh.add_element([4,2], "poinçon", "m", 22 ,10)
mesh.add_element([4,1], "arba", "b", 22 ,10)
mesh.geom()

#Modele
f = fem.FEM_Model(mesh)
# bc
f.apply_bc([1,1,0],1)
f.apply_bc([1,1,0],3)
f.apply_distributed_load(1000, [1,4])
f.apply_distributed_load(1000, [4,3])
f.plot_forces(type = 'dist', pic = True)
f.solver_frame()
U, React, res = f.get_res()
f.U_table()
f.R_table()
f.plot_disp_f(dir='x', pic = True)
f.rapport()