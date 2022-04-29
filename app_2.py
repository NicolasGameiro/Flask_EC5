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

def cb(Load : int, nb_bolt : int) -> int :
    return Load/nb_bolt

def calcul_solive(h,l,e,p,P,Q):
    I = l*h**3/12 #Inertie de la poutre
    E = 15000 #Module d'Young du bois
    S = h*l #Section de la poutre
    q_els = (1.35*P + 1*Q)*e
    q_elu = (1.35*P + 1.5*Q)*e
    #ELS
    fleche = q_els*p**3/48/E/I
    #ELU
    stc = 0
    sf = q_elu/I*h/2
    sc = q_elu/S
    return stc, sf, sc, fleche


app = Flask(__name__)
app.config['SECRET_KEY'] = "secret"

#Create a form class
class SoliveForm(FlaskForm):
    name = StringField("Load applied to the bolt :", validators = [DataRequired()])
    hauteur = IntegerField("Hauteur (cm) : ")
    largeur = IntegerField("Largueur (cm) : ")
    entraxe = DecimalField("Entraxe (cm) : ")
    portee = DecimalField("Portée (m) : ")
    P = DecimalField("Charge permanente (kN/m2) : ")
    Q = DecimalField("Charge variable (kN/m2) : ")
    submit = SubmitField("Calculer")
    
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
        data = []
        data.append(int(request.form.get('classe_bois')))
        h = int(request.form.get('hauteur'))
        l = int(request.form.get('largeur'))
        e = float(request.form.get('entraxe'))
        p = float(request.form.get('porte'))
        P = float(request.form.get('P'))
        Q = float(request.form.get('Q'))
        res = calcul_solive(h,l,e,p,P,Q)
        return render_template("solive.html", data = data, res = res)
        #stc, sf, sc, fleche = calcul_solive(h, l, e, p, c_p, c_v)
        #sigma = [stc, sf, sc]
    return render_template("solive.html")


if __name__ == "__main__":
    app.run(host ='0.0.0.0',debug=True)