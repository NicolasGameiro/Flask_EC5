# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, send_file, flash, request, url_for
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,DecimalField,RadioField,SelectField,IntegerField
from wtforms.validators import DataRequired
import plotly 
import plotly.express as px
import json
import pandas as pd
import numpy as np
import EC5
import sys, os
print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'\\src')
print(sys.path)

from src.gen_report import rapport

def cb(Load : int, nb_bolt : int) -> int :
    return Load/nb_bolt

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

    
class BoltForm(FlaskForm):
    nb_bolt = IntegerField("Nombre de vis : ")
    load = IntegerField("Charge appliquée sur les vis : ")
    submit = SubmitField("Calcul")
    

@app.route('/bolt',methods=['GET','POST'])
def bolt():
    load = 0
    nb_bolt = 0
    res = 0
    form = BoltForm()
    #Validate Form 
    if form.validate_on_submit():
        nb_bolt = form.nb_bolt.data
        load = form.load.data
        res = cb(load,nb_bolt)
        form.nb_bolt.data = ''
        form.load.data = ''
        flash('Le calcul a été lancé !')
    return render_template('bolt.html', res = res, load = load, nb_bolt = nb_bolt, form = form)


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
        b = request.form.get('type_bois')
        cs = request.form.get("classe_service")
        cq = request.form.get("classe_bois")
        q_elu, q_els = EC5.charge(bande, pp, G, Q)
        sig = EC5.calcul_solive(h, l, bande, p, q_elu, q_els)
        res = EC5.calcul_taux_trav(sig[0], sig[1], sig[2], b, cs, cq)
        return render_template("solive.html", q = q_elu ,sig = sig,  res = res)
        flash('calcul lancé !')
        #stc, sf, sc, fleche = calcul_solive(h, l, e, p, c_p, c_v)
        #sigma = [stc, sf, sc]
    return render_template("solive.html")

@app.route('/panne')
def panne():
    return render_template("panne.html")

@app.route('/node', methods=['GET','POST'])
def node():
    if request.method == "POST" and request.form['button'] == "add_node" : 
            x = request.form.get("x")
            y = request.form.get("y")
            z = request.form.get("z")
            NL = [[x, y, z]]
            flash('Le noeud a été ajouté avec succès !')
            return render_template("node.html", NL = NL)
    elif request.method == "POST" and request.form['button'] == "del_node" : 
            NL =  [[]] 
            flash('Le noeud a été supprimé avec succès !')
            return render_template("node.html", NL = NL)
    elif request.method == "POST" and request.form['button'] == "add_elem" : 
            n1 = request.form.get("n1")
            n2 = request.form.get("n2")
            EL = [[n1, n2]]
            flash("L'element a été ajouté avec succès !")
            return render_template("node.html", EL = EL)
    elif request.method == "POST" and request.form['button'] == "del_elem" : 
            EL =  [[]] 
            flash("L'element a été supprimé avec succès !")
            return render_template("node.html", EL = EL)
    else : 
        return render_template("node.html")

@app.route('/Report', methods=['POST'])
def report():
    if request.method == "POST" : 
        output = request.form.to_dict()
        rapport(output)
    return render_template("solive.html")


if __name__ == "__main__":
    app.run(host ='0.0.0.0',debug=True)