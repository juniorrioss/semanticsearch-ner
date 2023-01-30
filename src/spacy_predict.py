import spacy

nlp = spacy.load("pt_core_news_lg", disable=["tagger", "attribute_ruler", "lemmatizer"])

doc = nlp(
    "A Dona Maria lavou a calçada com água e tomou um esporro do senhor Jamil que não gostou nada da situação!"
)
print([(w.text, w.pos_, w.ent_iob_, w.ent_type_) for w in doc])
