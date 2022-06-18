import matplotlib.pyplot as plt

from mesh import Mesh
from model import FEM_Model

def test_3d():
    m1 = Mesh(dim=3)
    p = 6.5
    h = 2
    h_mur = 2.5
    L = 6
    m1.add_node([0, 0, 0])
    m1.add_node([0, 0, h_mur])
    m1.add_node([0, p / 2, h_mur + h])
    m1.add_node([0, p, h_mur])
    m1.add_node([0, p, 0])
    m1.add_node([L, p / 2, h_mur + h])
    m1.add_element([1, 2], "poteau", "k", 15, 15)
    m1.add_element([2, 3], "arba", "r", 22, 12)
    m1.add_element([3, 4], "arba", "r", 22, 12)
    m1.add_element([4, 2], "entrait", "b", 22, 10)
    m1.add_element([4, 5], "poteau", "k", 15, 15)
    m1.add_element([3, 6], "panne faitiere", 'g', 22, 12)
    # m1.geom()
    m1.node_table()
    # plt.show()
    f = FEM_Model(m1)
    f.apply_distributed_load(1, [3, 6])
    f.plot_forces3D(type='nodal')
    f.apply_bc([1, 1, 1, 0, 0, 0], 1)
    f.apply_bc([1, 1, 1, 1, 1, 1], 5)
    f.apply_bc([1, 1, 1, 1, 1, 1], 6)
    f.solver_frame()
    f.plot_disp_f_3D(dir="x")
    plt.show()
    f.U_table()
    f.R_table()
    return


def validation_3d():
    m1 = Mesh(dim=3)
    m1.add_node([0, 0, 0])
    m1.add_node([0, 0, 1])
    m1.add_node([0, 1, 1])
    # m1.add_node([1, 1, 1])
    m1.add_element([1, 2], "poteau", "k", 10, 10)
    m1.add_element([2, 3], "arba", "r", 10, 10)
    # m1.add_element([3, 4], "arba", "r", 10, 10)
    m1.node_table()
    f = FEM_Model(m1)
    # f.apply_distributed_load(1, [2, 3])
    f.apply_load([0, 0, -0.1, 0, 0, 0], 3)
    f.plot_forces3D(type='nodal')
    f.apply_bc([1, 1, 1, 1, 1, 1], 1)
    # f.apply_bc([1, 1, 1, 1, 1, 1], 4)
    f.solver_frame()
    f.plot_disp_f_3D(dir="sum")
    plt.show()
    f.U_table()
    f.R_table()
    return


def validation_2d():
    mesh = Mesh(2, [], [], debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])  # inches
    mesh.add_node([10, 10])  # inches
    mesh.add_element([1, 2], "barre", "b", 15, 15, 5)
    mesh.add_element([2, 3], "barre", "b", 15, 15, 5)
    # mesh.geom()

    f = FEM_Model(mesh)
    f.apply_distributed_load(10, [6, 11])
    f.apply_bc([1, 1, 1], 1)
    # f.apply_bc([0,1,0],3)
    # print(f.get_bc())
    f.plot_forces(type='dist', pic=False)
    f.solver_frame()
    # f.plot_disp_f(dir='x', pic = True)
    # f.plot_disp_f(dir='y' , pic = True)
    f.plot_disp_f(dir='sum', pic=True)
    # f.plot_disp_f_ex()
    S = f.stress()
    f.plot_stress(scale=1e2, r=100, s='sf', pic=False, path="./")
    f.S_table()
    f.U_table()
    f.R_table()


def test_2d():
    mesh = Mesh(2, [], [], debug=False)
    p = 6.5
    h = 2.5
    mesh.add_node([0, 0])
    mesh.add_node([p / 2, 0])
    mesh.add_node([p, 0])
    mesh.add_node([p / 2, h])
    mesh.add_node([p / 4, h / 2])
    mesh.add_node([3 * p / 4, h / 2])
    mesh.add_element([1, 2], "entrait", "r", 22, 10)
    mesh.add_element([2, 3], "entrait", "r", 22, 10)
    mesh.add_element([3, 6], "arba", "g", 20, 8)
    mesh.add_element([6, 4], "arba", "g", 20, 8)
    mesh.add_element([4, 5], "arba", "g", 20, 8)
    mesh.add_element([5, 1], "arba", "g", 20, 8)
    mesh.add_element([4, 2], "poinçon", "b", 10, 10)
    mesh.add_element([2, 5], "jdf", "m", 10, 10)
    mesh.add_element([2, 6], "jdf", "m", 10, 10)
    mesh.geom()
    # mesh.node_table()

    f = FEM_Model(mesh)
    # f.apply_load([0,-1000,0],4)
    f.apply_bc([1, 1, 1], 1)
    f.apply_bc([1, 1, 1], 5)
    f.apply_distributed_load(2000, [1, 4])
    f.apply_distributed_load(2000, [4, 3])
    f.plot_forces(type='dist', pic=True)
    f.solver_frame()
    U, React, res = f.get_res()
    f.plot_disp_f(dir='x', scale=1e3)
    # f.plot_disp_f(dir='y' , scale = 1e3, pic = True)
    # f.plot_disp_f(dir='sum', scale = 1e3, pic = True)
    # f.plot_disp_f_ex()
    f.U_table()
    f.R_table()
    # f.stress()
    # f.rapport()
    return


def test_cantilever():
    mesh = Mesh(2, [], [], debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([1, 0])
    mesh.add_element([1, 2], "entrait", "r", 22, 10)
    f = FEM_Model(mesh)
    f.apply_load([0, -1000, 0], -1)
    f.apply_bc([1, 1, 1], 1)
    f.plot_forces(type='dist', pic=False)
    f.solver_frame()
    # f.plot_disp_f_ex(scale=1e2)
    f.plot_disp_f(scale=1e4, dir='y')
    f.plot_stress(scale=1e4, r=100, s='sf', pic=False, path="./")
    f.S_table()
    return

if __name__ == "__main__":
    test_2d()
    plt.show()