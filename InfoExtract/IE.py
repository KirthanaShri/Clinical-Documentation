import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

import spacy
import scispacy

import en_core_sci_sm
import en_core_sci_lg

import os

med7 = spacy.load("en_core_med7_lg")
print(med7.pipe_labels['ner'])

options = {'ents': med7.pipe_labels['ner']}

def create_df(path):
    file_ents = []
    drugs=[]
    dosages = []
    df =pd.DataFrame()
    #  Loop for reading a document file and doing string manipulation.
    r=0
    for file in os.listdir(path):
        with open(path + file, 'r', encoding='utf8') as f:
            text = f.readlines()
        str1 = ""
        for line in text:
            str1 += line

        doc = med7(str1)

        for ent in doc.ents:
            df.loc[r, 'File'] = str(file)
            df.loc[r, 'Text'] = ent.text
            df.loc[r, 'Class'] = ent.label_
            r+=1

    return df



path = "/Users/kirthanashri/PycharmProjects/DOCU CLINICAL CLASS/CLINICAL DATA/data1/"
df = create_df(path)
print(df)

#Load specific model and pass text through
nlp = en_core_sci_lg.load()
doc = nlp(text)


# from medacy.model.model import Model
#
# model = Model.load_external('medacy_model_clinical_notes')
# annotation = model.predict("The patient was prescribed 1 capsule of Advil for 5 days.")
# print(annotation)



