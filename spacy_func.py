import spacy

class spacy_func(object):
    w2v_model=None

    def __call__(self,x):
#         x=x.split(' ')

        if not spacy_func.nlp:
            nlp=spacy.load('en')
#         vector = np.zeros(300)
#         for word in x:
#             try:
#                 vector +=w2v_model[word]
#             except:
#                 pass
#         return vector
        return spacy_func.nlp(x).ents