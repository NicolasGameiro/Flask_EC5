# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:27:25 2022

@author: ngameiro
"""

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import tracer_charpente_v5 as tc

fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
matplotlib.use("TkAgg")

sg.theme('Dark')
SYMBOL_UP =    '▲'
SYMBOL_DOWN =  '▼'

def collapse(layout, key):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key))

def fig_maker(ep_dalle, largeur_batiment, ep_mur1, h_mur1):
    plt.clf()
    plt.close()
    fig, ax1 = plt.subplots()
    try : 
        tc.walls(ax1, ep_dalle, largeur_batiment, ep_mur1, h_mur1, "sym")
    except : 
        pass
    try : 
        tc.charpente(axe_ferme , h_archi , offset_sablier , b_sablier , h_sablier , debord_gauche , debord_droite , ep_couv , ep_chevron)
    except : 
        pass
    ax1.axis('equal')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return plt.gcf()

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')

# Define the window layout
# maconnerie
frame_1 = [sg.Frame(layout=[    
    [sg.Text("épaisseur dalle (cm) : "), sg.Input(s=6,enable_events=True, default_text="20", key="ep_dalle"),
     sg.Text("largeur batiement (cm) : "), sg.Input(s=6,enable_events=True, default_text="800", key="l_bat")],
    [sg.Text("hauteur mur 1 (cm) : "), sg.Input(s=6,enable_events=True, default_text="200", key="h_mur1"),
     sg.Text("epaisseur mur 1 (cm) : "), sg.Input(s=6,enable_events=True, default_text="20", key="ep_mur1")],
    [sg.Text("hauteur mur 2 (cm) : "), sg.Input(s=6,enable_events=True, default_text="200", key="h_mur2"),
     sg.Text("epaisseur mur 2 (cm) : "), sg.Input(s=6,enable_events=True, default_text="20", key="ep_mur2")],
    ], title='Maçonnerie',title_color='red', relief=sg.RELIEF_SUNKEN)]

frame_2 = [sg.Frame(layout=[    
    [sg.Text("axe ferme (cm) : "), sg.Input(s=6,enable_events=True, default_text="400", key="axe"),
     sg.Text("hauteur architecture (cm) : "), sg.Input(s=6,enable_events=True, default_text="700", key="h_archi")],
    [sg.Text("offset sabliere (cm) : "), sg.Input(s=6,enable_events=True, default_text="10", key="offset"),
     sg.Text("hauteur sabliere (cm) : "), sg.Input(s=6,enable_events=True, default_text="10", key="h_sab"),
     sg.Text("base sabliere (cm) : "), sg.Input(s=6,enable_events=True, default_text="10", key="b_sab")],
     [sg.Text("debord gauche (cm) : "), sg.Input(s=6,enable_events=True, default_text="40", key="deb_1"),
      sg.Text("debord droite (cm) : "), sg.Input(s=6,enable_events=True, default_text="40", key="deb_2")],
     [sg.Text("epaisseur chevron (cm) : "), sg.Input(s=6,enable_events=True, default_text="20", key="ep_chevron"),
      sg.Text("epaisseur couverture (cm) : "), sg.Input(s=6,enable_events=True, default_text="20", key="ep_couv")],
    ], title='Charpente',title_color='red', relief=sg.RELIEF_SUNKEN)]

frame_3 = [sg.Frame(layout=[    
    [sg.Text("rayon goutiere (cm) : "), sg.Input(s=6,enable_events=True, default_text="400", key="axe")],
    ], title='Couverture',title_color='red', relief=sg.RELIEF_SUNKEN)]

layout_l = [ frame_1 ,
            frame_2 ,
            frame_3 ,
    [sg.Button("Run calculation"),
     sg.Button("sauvegarder image")],
    [sg.Checkbox('Ferme symétrique', default=True, key="sym")],
]

layout_r = [
    [sg.Canvas(key="-CANVAS-")],
]

section1 = [[sg.Input('Input sec 1', key='-IN1-')],
            [sg.Input(key='-IN11-')],
            [sg.Button('Button section 1',  button_color='yellow on green'),
             sg.Button('Button2 section 1', button_color='yellow on green'),
             sg.Button('Button3 section 1', button_color='yellow on green')]]

section2 = [[sg.I('Input sec 2', k='-IN2-')],
            [sg.I(k='-IN21-')],
            [sg.B('Button section 2',  button_color=('yellow', 'purple')),
             sg.B('Button2 section 2', button_color=('yellow', 'purple')),
             sg.B('Button3 section 2', button_color=('yellow', 'purple'))]]

layout = [[sg.MenubarCustom([['File', ['Exit']], ['Edit', ['Edit Me', ]]], k='-CUST MENUBAR-',p=0)],
          [sg.T('Générateur de ferme', font='_ 16', justification='c', expand_x=True)],
          [sg.Col(layout_l), sg.Col(layout_r)],
          #### Section 1 part ####
        [sg.T(SYMBOL_DOWN, enable_events=True, k='-OPEN SEC1-', text_color='yellow'), sg.T('Section 1', enable_events=True, text_color='yellow', k='-OPEN SEC1-TEXT')],
        [collapse(section1, '-SEC1-')],
        #### Section 2 part ####
        [sg.T(SYMBOL_DOWN, enable_events=True, k='-OPEN SEC2-', text_color='purple'),
         sg.T('Section 2', enable_events=True, text_color='purple', k='-OPEN SEC2-TEXT')],
        [collapse(section2, '-SEC2-')],
        ]

#[sg.Radio('My first Radio!     ', "RADIO1", default=True, size=(10,1)), sg.Radio('My second Radio!', "RADIO1")]

# Create the form and show it without the plot
window = sg.Window(
    "Matplotlib Single Graph",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
)

fig_agg = None
opened1, opened2 = True, True

while True:
    # Add the plot to the window
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "Run calculation":
        ep_dalle = float(values['ep_dalle'])
        h_mur1 = float(values['h_mur1'])
        h_mur2 = float(values['h_mur2'])
        largeur_batiment, ep_mur1 = 800 , 20
        print(ep_dalle,h_mur1, h_mur2)
        #delete_figure_agg(window["-CANVAS-"].TKCanvas, fig)
        if fig_agg is not None:
            delete_fig_agg(fig_agg)
        fig = fig_maker(ep_dalle, largeur_batiment, ep_mur1, h_mur1)
        fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
        window.Refresh()
    if event == "sauvegarder image" : 
        fig.savefig('schema_ferme.png', format='png', dpi=200)
    
    if event.startswith('-OPEN SEC1-'):
        opened1 = not opened1
        window['-OPEN SEC1-'].update(SYMBOL_DOWN if opened1 else SYMBOL_UP)
        window['-SEC1-'].update(visible=opened1)

    if event.startswith('-OPEN SEC2-'):
        opened2 = not opened2
        window['-OPEN SEC2-'].update(SYMBOL_DOWN if opened2 else SYMBOL_UP)
        window['-OPEN SEC2-CHECKBOX'].update(not opened2)
        window['-SEC2-'].update(visible=opened2)

window.close()