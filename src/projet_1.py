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
    
def rotation_matrix_from_vectors(vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    RR = np.identity(12)
    vec1 = [1,0,0]
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    RR[0:3,0:3] = rotation_matrix
    RR[3:6,3:6] = rotation_matrix
    RR[6:9,6:9] = rotation_matrix
    RR[9:12,9:12] = rotation_matrix
    print(RR)
    return rotation_matrix

def pv(origin,vec, c= 'k') : 
    plt.quiver( origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], color = c)
    

if __name__ == "__main__":
    """
    with open('materiel.json') as json_data:
        data_dict = json.load(json_data)
        print(data_dict["Tuile mecanique"])
        
    data_dict["tuile plate"] = {"poids" : 0.4, "prix" : 35}
        
    with open('json_data.json', 'w') as outfile:
        json.dump(data_dict, outfile)
    """
    
    ori = [0, 0, 0]
    vec1 = [1,0,0]
    vec2 = [1,1,1]
    
    R = rotation_matrix_from_vectors(vec2)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # plt.gca(projection='3d')
    ### Trace les efforts
    pv(ori,[1,0,0], 'r')
    pv(ori,[0,1,0], 'g')
    pv(ori,[0,0,1], 'b')
    vec3 = R*vec1
    pv(ori,vec3)
    ax.set_title("Structure")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    ax.view_init(elev=20., azim=-20.)
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 2)
    plt.tight_layout()
    print(R)