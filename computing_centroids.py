
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

def compute_centroids(w2v, terms, words_frequency):  
    
    l = sum(words_frequency.values())

    a = 0.001
    for key in tqdm(words_frequency.keys()):
        words_frequency[key] = a / (a + (words_frequency[key] / l))

    def findcentroid(text, model):
        vecs = [model.wv[w] * words_frequency[w] for w in text if w in model.wv]
        
        # Additional debug check
        if not vecs:
            print("Empty vector list")
        vecs = [v for v in vecs if len(v) > 0]
        centroid = np.mean(vecs, axis=0)
        centroid = centroid.reshape(1, -1)
        return centroid


    centroid = findcentroid(terms, w2v)
    return centroid