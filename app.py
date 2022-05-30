# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import base64
from io import BytesIO
from matplotlib.figure import Figure

from flask import Flask, render_template, flash, request#, url_for, send_file
import EC5
import src.Code_FEM_v4 as fem
import sys, os
import json
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'\\src')
print(sys.path)

from src.gen_report import rapport

mesh = fem.Mesh(2,[[0,0],[1,0]], [[1,2]])
f = fem.FEM_Model(mesh)
with open('src/materiel.json') as json_data:
    data_dict = json.load(json_data)


#taux de travail
classe_de_service = {"classe 1" : 0.8 , 
                     "classe 2" : 0.7 ,
                     "classe 3 " : 0.6}

# Categorie charge d'exploitation (en kN/m2)
categorie_charge_exploitation = { "A" : 1.5,
                                 "B" : 2.5,
                                 "C" : 2.5,
                                 "D" : 5,
                                 "E" : 7.5}

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0    


@app.route('/assemblage',methods=['GET','POST'])
def assemblage():
    return render_template('assemblage.html')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/solive', methods=['GET','POST'])
def solive():
    if request.method == "POST" : 
        h = int(request.form.get('hauteur'))/100
        l = int(request.form.get('largeur'))/100
        bande = float(request.form.get('entraxe'))/100
        p = float(request.form.get('porte'))/100
        pp = 0.4
        G = float(request.form.get('G'))
        Q = float(request.form.get('Cat_charge'))
        cs = request.form.get("classe_service")
        cq = request.form.get("classe_bois")
        q_elu, q_els = EC5.charge(bande, pp, G, Q)
        sig = EC5.calcul_solive(h, l, bande, p, q_elu, q_els)
        res = EC5.calcul_taux_trav(sig[0], sig[1], sig[2], cs, cq)
        return render_template("solive.html", q = q_elu ,sig = sig,  res = res)
        flash('calcul lancé !')
        #stc, sf, sc, fleche = calcul_solive(h, l, e, p, c_p, c_v)
        #sigma = [stc, sf, sc]
    return render_template("solive.html")

@app.route('/panne')
def panne():
    return render_template("panne.html")

@app.route('/charge', methods=['GET','POST'])
def charge():
    if request.method == "POST" and request.form['button'] == "add" :
        nom = request.form.get("nom")
        poids = float(request.form.get("poids"))
        prix = float(request.form.get("prix"))
        coloris = request.form.get("coloris")
        finition = request.form.get("finition")
        data_dict[nom] = {"poids" : poids, "prix" : prix, "coloris" : coloris, "finition": finition}
    elif request.method == "POST" and request.form['button'] == "save" :
        with open('src/materiel.json', 'w') as outfile:
            json.dump(data_dict, outfile)
        flash('materiau ajouté')
    return render_template("charge.html", materiel = data_dict)

@app.route('/node', methods=['GET','POST'])
def node():
    #render_template("node.html", NL = mesh.node_list)
    if request.method == "POST" and request.form['button'] == "add_node" : 
        x = int(request.form.get("x"))
        y = int(request.form.get("y"))
        f.mesh.add_node([x, y])
        flash('Le noeud a été ajouté avec succès !')
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list , BC = f.get_bc() , LL = f.load)
    elif request.method == "POST" and request.form['button'] == "del_node" : 
        x = int(request.form.get("x"))
        y = int(request.form.get("y"))
        f.mesh.del_node([x, y])
        flash('Le noeud a été supprimé avec succès !')
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list , BC = f.get_bc() , LL = f.load)
    elif request.method == "POST" and request.form['button'] == "add_elem" : 
        n1 = int(request.form.get("n1"))
        n2 = int(request.form.get("n2"))
        f.mesh.add_element([n1, n2])
        flash("L'element a été ajouté avec succès !")
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list , BC = f.get_bc() , LL = f.load)
    elif request.method == "POST" and request.form['button'] == "del_elem" : 
        n1 = int(request.form.get("n1"))
        n2 = int(request.form.get("n2"))
        f.mesh.del_element([n1, n2])
        flash("L'element a été supprimé avec succès !")
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list , BC = f.get_bc() , LL = f.load)
    elif request.method == "POST" and request.form['button'] == "geom" : 
        flash("node list = "+str(f.mesh.node_list) + "et element list = " + str(f.mesh.element_list))
        f.mesh.geom(pic = True)
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list,  BC = f.get_bc() , LL = f.load, im = 'geom.png')
    elif request.method == "POST" and request.form['button'] == "run" : 
        flash("node list = "+str(f.mesh.node_list) + "et element list = " + str(f.mesh.element_list))
        f.plot_forces(pic = True)
        f.plot_disp_f(dir='x', pic = True)
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list , BC = f.get_bc() , LL = f.load, im = 'res_x.png')
    elif request.method == "POST" and request.form['button'] == "add_cl" : 
        n = int(request.form.get("node"))
        ux = request.form.get("ux")
        uy = request.form.get("uy")
        rz = request.form.get("rz")
        if ux is None : 
            ux = 0
        if uy is None : 
            uy = 0
        if rz is None : 
            rz = 0
        f.apply_load([0,-1000,0],4)
        f.apply_bc([int(ux),int(uy),int(rz)],n)
        flash("node " + str(n) + " has the following bc : " + str(ux) + ", " + str(uy) + ", " + str(rz)) 
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list, BC = f.get_bc() , LL = f.load)
    elif request.method == "POST" and request.form['button'] == "add_load" : 
        n = int(request.form.get("node"))
        fx = request.form.get("fx")
        fy = request.form.get("fy")
        mz = request.form.get("mz")
        f.apply_load([int(fx),int(fy),int(mz)],n)
        flash("node " + str(n) + " has the following load : Fx = " + str(fx) + ", Fy = " + str(fy) + ", Mz = " + str(mz)) 
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list, BC = f.get_bc() , LL = f.load)
    else : 
        return render_template("node.html", NL = f.mesh.node_list , EL = f.mesh.element_list, BC = f.get_bc() , LL = f.load)

@app.route('/Report', methods=['POST'])
def report():
    if request.method == "POST" : 
        output = request.form.to_dict()
        rapport(output)
    return render_template("solive.html")

@app.route('/hello', methods=['GET','POST'])
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"


if __name__ == "__main__":
    app.run(host ='0.0.0.0',debug=True)