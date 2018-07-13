import collections
import scipy.sparse as sp
import numpy as np
def tfidf(docs):
    all_words = set([a for a in " ".join(docs).split(" ") if a != ""])
    all_words_dict = {k:i for i,k in enumerate(all_words)}
    word_counts = [collections.Counter([a for a in d.split(" ") if a != ""]) for d in docs]
    
    # construct term frequency matrix in COO form
    data = [a for wc in word_counts for a in wc.values()]
    rows = [i for i,wc in enumerate(word_counts) for a in wc.values()]
    cols = [all_words_dict[k] for wc in word_counts for k in wc.keys()]
    X = sp.coo_matrix((data, (rows,cols)), (len(docs), len(all_words)))
    
    # compute IDF and TFIDF terms
    idf = np.log(float(len(docs))/np.asarray((X > 0).sum(axis=0))[0])
    return X * sp.diags(idf), list(all_words)