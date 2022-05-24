# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 00:18:36 2022

@author: ngameiro

#TODO
- calcul des appuies

"""

import numpy as np
from prettytable import PrettyTable as pt

def calcul_solive(h : float, l : float, e : float, p : float, q_elu : float, q_els : float) :
    I = l*h**3/12 #Inertie de la poutre (en m4)
    E = 11000e3 #Module d'Young du bois (en Pa)
    S = h*l #Section de la poutre (m2)
    M = q_elu*1000*p**2/8 #Moment autour de l'axe fort
    T = q_elu*p*1000
    #ELS
    f = np.round(5*p**4/384/E/I*100,2) # pas de prise en compte des efforts tranchants (en cm)
    #ELU
    stc = np.round(q_elu/1e6/l,2) #MPa
    sf = np.round(M/I*h/2/1e6,2) #MPa
    sv = np.round(3/2*T/S/1e6,2) #MPa
    return stc, sf, sv, f

def calcul_appui():
    return 

def charge_neige(pente):
    """ fonction qui calcule le chargement surfacique de neige en fonction de la pente
    
    :param pente: pente de la toiture en degre
    :type pente: float
    :return: charge de neige en kN/m2
    :rtype: float

    """
    if pente <= 30:
        coeff_forme = 0.8
    elif pente <= 60 :
        coeff_forme = 0.8*(60 - pente)/30
    else : 
        coeff_forme = 0
    sk = 0.45 #kN/m2 (région A1)
    Ct = 1 #Coefficient thermique
    Ce = 1 #Coefficient d'exposition
    s = sk*Ct*Ce*coeff_forme
    return s

def charge_vent(pente):
    return

def calcul_panne(h : float, l : float, e : float, p : float, q_elu : float, q_els : float, pente : float =0) :
    a = np.deg2rad(pente)
    Wy = l*h**2/12 #Inertie de la poutre (en m4)
    Wz = h*l**2/12 #Inertie de la poutre (en m4)
    E = 11000e3 #Module d'Young du bois (en Pa)
    S = h*l #Section de la poutre (m2)
    qz = np.cos(a)*q_elu
    qy = np.sin(a)*q_elu
    My = qz*1000*p**2/8 #Moment autour de l'axe fort
    Mz = qy*1000*p**2/8 #Moment autour de l'axe faible
    T = q_elu*p*1000
    #ELS
    f = np.round(5*p**4/384/E/Wz/h*2*100,2) # pas de prise en compte des efforts tranchants (en cm)
    #ELU
    stc = np.round(q_elu/1e6/l,2) #MPa
    sf = np.round((Mz/Wz + My/Wy)/1e6,2) #MPa
    sv = np.round(3/2*T/S/1e6,2) #MPa
    return stc, sf, sv, f

def calcul_arba(h : float, b : float, q_p : float, L : float, q_neige : float, q_vent : float, pente : float =0) :
    a = np.deg2rad(pente)
    S= h*b
    I = b*h**3/12
    qf = q_neige*np.cos(a)**2 + q_vent + q_p*np.cos(a)
    qc= q_neige*np.cos(a)*np.sin(a) + q_p*np.sin(a)
    Mf = qf*L**2/8
    V = qf*L/2
    C = qc*L
    stc = C/S
    sf = Mf/I*h/2
    sv = V/S*3/2
    
    return stc, sf, sv

def calcul_taux_trav(stc,sf,sv,cs,cq) : #,classe_s,classe_q,type_b) : 
    # Valeurs caractéristiques
    # cq = "C24"
    # cs = "classe 1"
    fm_k = classe_qualite[cq]['fm,k']
    ft_k = classe_qualite[cq]['ft,0,k']
    fc_k = classe_qualite[cq]['fc,0,k']
    fv_k = classe_qualite[cq]['fv,k']
    # Valeurs Réduites
    k_mod = classe_de_service[cs]
    if 'C' in cq : 
        gamma_m = 1.3
    elif 'GL' in cq : 
        gamma_m = 1.25
    fm_d = fm_k*k_mod/gamma_m
    ft_d = ft_k*k_mod/gamma_m
    fc_d = fc_k*k_mod/gamma_m
    fv_d = fc_k*k_mod/gamma_m
    taux_travail_t = np.round(stc/ft_d*100,1)
    taux_travail_m = np.round(sf/fm_d*100,1)
    taux_travail_v = np.round(sv/fv_d*100,1)
    return taux_travail_t,taux_travail_m,taux_travail_v
    

#taux de travail

classe_de_service = {"classe 1" : 0.8 , 
                     "classe 2" : 0.7 ,
                     "classe 3 " : 0.6}

bois = {'BM' : 1.3,
        'LC' : 1.25}

classe_qualite = {"C24" : { "fm,k" : 24,
                            "ft,0,k" : 14,
                            "ft,90,k" : 0.4,
                            "fc,0,k" : 21,
                            "fc,90,k" : 2.5,
                            "fv,k" : 2.5,
                            "E0,m" : 11000,
                            "Gm" : 690,
                            "rho_m" : 420 #kg/m3
                           },
                  "C22" : { "fm,k" : 22,
                            "ft,0,k" : 14,
                            "ft,90,k" : 0.4,
                            "fc,0,k" : 21,
                            "fc,90,k" : 2.5,
                            "fv,k" : 2.5,
                            "E0,m" : 11000,
                            "Gm" : 690,
                            "rho_m" : 420 #kg/m3
                           }
                  }

# Categorie charge d'exploitation (en kN/m2)
categorie_charge_exploitation = { "A" : 1.5,
                                 "B" : 2.5,
                                 "C" : 2.5,
                                 "D" : 5,
                                 "E" : 7.5}

def charge(bande : float, pp : float,G : float ,Q : float ,W : float =0 ,S :float =0) -> float: 
    """

    Parameters
    ----------
    bande : float
        bande de chargement (m).
    pp : float
        Charge lineique du poids propre (kN/m).
    G : float
        Charge surfacique permanente (kN/m2).
    Q : float
        Charge surfacique variable (kN/m2).
    W : float
        Charge surfacique de vent (kN/m2). The default is 0 : float.
    S : float
        Charge surfacique de neige (kN/m2). The default is 0 : float.

    Returns
    -------
    float
        la charge linéique sur la poutre.

    """
    q_elu = np.round(bande*(1.35*G + 1.5*Q + W + S) + pp,2)
    q_els = np.round(bande*(G + Q + W + S) + pp, 2)
    return q_elu, q_els

def main():
    k_mod = classe_de_service['classe 1']
    gamma_m = bois['BM']
    fm = classe_qualite["C24"]["fm,k"]
    rho = classe_qualite["C24"]["rho_m"]
    h = 0.22
    l = 0.1
    p = 4
    bande = 0.5 
    pp = rho*h*l/100 #kN/m
    G = 0.4
    Q = 1
    q_elu, q_els = charge(bande,pp,G,Q)
    stc, sf, sv, f = calcul_solive(h, l, bande, p, q_elu, q_els)
    t1,t2,t3 = calcul_taux_trav(stc,sf,sv,'BM','classe 1','C24')
    print("-----")
    print("taux de travail : ")
    print("traction/compression = ", t1, "%")
    print("flexion = ", t2, "%")
    print("cisaillement = ", t3, "%")
    
    tb = pt()#Add headers
    tb.field_names = ["ID","Name", "Value","Unit"]#Add rows
    tb.add_row([1,"charge ELU", q_elu, "kN/m"])
    tb.add_row([2,"charge ELS", q_els, "kN/m"])
    tb.add_row([3,"contrainte t/c ", stc, "MPa"])
    tb.add_row([4,"contrainte flexion ",sf, "MPa"])
    tb.add_row([5,"contrainte cisaillement",sv, "MPa"])
    tb.add_row([6,"fleche ",f, "cm"])
    print(tb)
    

if __name__ == "__main__" :
    h = 0.22
    b = 0.08
    q_p = 400
    L = 4
    q_neige = 144
    q_vent = 0
    pente = 57.74
    stc, sf, sv = calcul_arba(h, b, q_p, L, q_neige, q_vent, pente )
    print(stc, sf , sv)