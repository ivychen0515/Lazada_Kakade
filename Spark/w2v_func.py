import gensim
import numpy as np


# from identity import Identity

# class F(object):                          
#     identity = None

#     def __call__(self, x):
#         if not F.identity:
#             F.identity = Identity()
#         return F.identity.identity(x)
# class Identity(object):                   
#     def __getstate__(self):
#         raise NotImplementedError("Not serializable")

#     def identity(self, x):
#         return x
# def w2v_func(col):
#     vector = np.zeros(300)
#     for word in title:
#         try:
#             vector +=w2v_model[word]
#         except:
#             pass
#     return vector  

class w2v_func(object):
    w2v_model=None

    def __call__(self,x):
#         x=x.split(' ')

        if not w2v_func.w2v_model:
            w2v_model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
#         vector = np.zeros(300)
#         for word in x:
#             try:
#                 vector +=w2v_model[word]
#             except:
#                 pass
#         return vector
        return w2v_model.word_vec(x)