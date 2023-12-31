import spacy
import scispacy
import en_core_sci_lg
nlp = spacy.load("en_core_sci_lg")
text = """spaCy is an open-source software library for advanced natural language processing, 
written in the programming languages Python and Cython. The library is published under the MIT license
and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
doc = nlp(text)
print(doc.ents)
