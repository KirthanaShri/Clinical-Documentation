import scispacy
import spacy

#Core models
import en_core_sci_sm
import en_core_sci_lg

#NER specific models
import en_ner_craft_md
import en_ner_bc5cdr_md
import en_ner_jnlpba_md
import en_ner_bionlp13cg_md

#Tools for extracting & displaying data
from spacy import displacy
import pandas as pd

#Read in csv file
meta_df = pd.read_csv("/content/sample.csv")

#Pick specific abstract to use (row 0, column "abstract")
text = meta_df.loc[0, "abstract"]

#Load specific model and pass text through
nlp = en_core_sci_lg.load()
doc = nlp(text)

#Display resulting entity extraction
displacy_image = displacy.render(doc, jupyter=True,style='ent')

# Load the models
nlp_cr = en_ner_craft_md.load()
nlp_bc = en_ner_bc5cdr_md.load()
nlp_bi = en_ner_bionlp13cg_md.load()
nlp_jn = en_ner_jnlpba_md.load()

def add_cr(abstractList, doiList):
    i = 0
    table= {"doi":[], "Entity":[], "Class":[]}
    for doc in nlp_cr.pipe(abstractList):
        doi = doiList[i]
        for x in doc.ents:
          table["doi"].append(doi)
          table["Entity"].append(x.text)
          table["Class"].append(x.label_)
        i +=1
    return table

def add_bc(abstractList, doiList):
    i = 0
    table= {"doi":[], "Entity":[], "Class":[]}
    for doc in nlp_bc.pipe(abstractList):
        doi = doiList[i]
        for x in doc.ents:
          table["doi"].append(doi)
          table["Entity"].append(x.text)
          table["Class"].append(x.label_)
        i +=1
    return table

def add_bi(abstractList, doiList):
    i = 0
    table= {"doi":[], "Entity":[], "Class":[]}
    for doc in nlp_bi.pipe(abstractList):
        doi = doiList[i]
        for x in doc.ents:
          table["doi"].append(doi)
          table["Entity"].append(x.text)
          table["Class"].append(x.label_)
        i +=1
    return table

#Read in file
meta_df = pd.read_csv("/content/sample.csv")

#Sort out blank abstracts
df = meta_df.dropna(subset=['abstract'])

#Create lists
doiList = df['doi'].tolist()
abstractList = df['abstract'].tolist()

#Add all entity value pairs to table (run one at a time, each ones takes ~20 min)
table = add_cr(abstractList, doiList)

# table = add_bc(abstractList, doiList)

# table = add_bi(abstractList, doiList)

# table = add_jn(abstractList, doiList)

#Turn table into an exportable CSV file (returns normalized file of entity/value pairs)
trans_df = pd.DataFrame(table)
trans_df.to_csv ("Entity_pairings.csv", index=False)

def add_jn(abstractList, doiList):
    i = 0
    table= {"doi":[], "Entity":[], "Class":[]}
    for doc in nlp_jn.pipe(abstractList):
        doi = doiList[i]
        for x in doc.ents:
          table["doi"].append(doi)
          table["Entity"].append(x.text)
          table["Class"].append(x.label_)
        i +=1
    return table

#Read in file
meta_df = pd.read_csv("/content/sample.csv")

#Sort out blank abstracts
df = meta_df.dropna(subset=['abstract'])

#Create lists
doiList = df['doi'].tolist()
abstractList = df['abstract'].tolist()

#Add all entity value pairs to table (run one at a time, each ones takes ~20 min)
table = add_cr(abstractList, doiList)

# table = add_bc(abstractList, doiList)

# table = add_bi(abstractList, doiList)

# table = add_jn(abstractList, doiList)

#Turn table into an exportable CSV file (returns normalized file of entity/value pairs)
trans_df = pd.DataFrame(table)
trans_df.to_csv ("Entity_pairings.csv", index=False)