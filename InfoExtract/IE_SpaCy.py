import spacy
import scispacy
import en_core_sci_lg
nlp = en_core_sci_lg.load()
text = """
Myeloid derived suppressor cells (MDSC) are immature 
myeloid cells with immunosuppressive activity. 
They accumulate in tumor-bearing mice and humans 
with different types of cancer, including hepatocellular 
carcinoma (HCC).
"""
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])
