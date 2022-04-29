# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 00:05:27 2022

@author: ngameiro
"""

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import time, datetime

def gen_a_doc(doc_name, preparation_module=None):
    """
    :param doc_name:
     It is a string, that contains a template name to render.
     Like if we have a report_template.docx than
     to the doc_name should be passed a string 'report_template'
     Nota Bene! There is to be a data-cooker. Called the same as the template
     For example: report_template.py
     And it has to contain a method context(), that returns
     a context dictionary for jinja2 rendering engine.
    :return:
    An file name located in TMP_DEST
    """
    if preparation_module is None:
        preparation_module = doc_name  # WOODOO MAGIC !!!!
    DOC_TEMPLATES_DIR = getattr(settings, "DOC_TEMPLATES_DIR", None)
    DOC_CONTEXT_GEN_DIR = getattr(settings, "DOC_CONTEXT_GEN_DIR", None)
    PROJECT_ROOT = getattr(settings, "PROJECT_ROOT", None)
    TMP_DEST = getattr(settings, "TMP_DEST", None)
    TMP_URL = getattr(settings, "TMP_URL", None)

    doc = DocxTemplate(os.path.join(PROJECT_ROOT, os.path.join(DOC_TEMPLATES_DIR, doc_name + ".docx")))
    print(os.path.join(PROJECT_ROOT, os.path.join(DOC_CONTEXT_GEN_DIR, preparation_module)))
    context_getter = import_module(preparation_module)
    context = getattr(context_getter, "context")()
    doc.render(context)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H:%M:%S")
    completeName = os.path.join(TMP_DEST, doc_name + st + ".docx")
    doc.save(completeName)
    return TMP_URL + doc_name + st + ".docx"

def rapport() : 
    doc = DocxTemplate("cctr_template.docx")
    
    myimage = InlineImage(doc, image_descriptor='schema_ferme.png', width=Mm(150), height=Mm(100))
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%d/%m/%y - %H:%M")
    short_st = datetime.datetime.fromtimestamp(ts).strftime("%d_%m_%y")
    
    context = { 'date' : st,
               'bois' :'C24',
               'var' : 30,
               'Image' : myimage}
    
    doc.render(context)
    doc.save("Rapport_" + short_st + ".docx")
    return print("Rapport genéré avec succès")

if __name__ == "__main__" : 
    rapport()