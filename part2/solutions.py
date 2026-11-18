from openTSNE import TSNE
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def analyse_embs(trained_emb_model, pos_words, neg_words):
    pos_embeddings = np.vstack([trained_emb_model.embed_word(pos_word) for pos_word in pos_words])
    neg_embeddings = np.vstack([trained_emb_model.embed_word(neg_word) for neg_word in neg_words])

    embeddings = np.vstack([pos_embeddings, neg_embeddings])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    km = KMeans(2)
    labels = km.fit_predict(embeddings)
    print('KMeans confustion matrix:')
    confusion_matrix = pd.DataFrame({"Positive":[(labels[:pos_embeddings.shape[0]] == 0).mean(), (labels[:pos_embeddings.shape[0]] == 1).mean()],
                                    "Negative":[(labels[pos_embeddings.shape[0]:] == 0).mean(), (labels[pos_embeddings.shape[0]:] == 1).mean()]},
                                    index=['Cluster 0', 'Cluster 1'])
    print(confusion_matrix)
    tsne = TSNE(
        perplexity=30,
        metric="cosine",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )


    tsne_emb = tsne.fit(embeddings)
    plt.scatter(tsne_emb[:pos_embeddings.shape[0], 0], tsne_emb[:pos_embeddings.shape[0], 1], s=1, label='Positive')
    plt.scatter(tsne_emb[pos_embeddings.shape[0]:, 0], tsne_emb[pos_embeddings.shape[0]:, 1], s=1, label='Negative')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def cos_similarity(u, v):  
    """
    Return the cosine similarity (a real number) of the two words in the embedding space of our model.
    """      
    if len(u.shape) == 1:
        u = u[np.newaxis, :]

    if len(v.shape) == 1:
        v = v[np.newaxis, :]


    return np.round((u @ v.T) / (np.linalg.norm(u, axis=1, keepdims=True) @ np.linalg.norm(v, axis=1, keepdims=True).T), 2)

def topk_ids(v, E, k=10):    
    """
    word embedding: a D-dimensional vector
    E: our model's embedding matrix as a numpy array of shape [V, D]
    k: how many nearest neighbours we want to find

    Return a list with the k word ids that are nearest to the input word embedding v
     in embedding space according to cosine similarity. 
    """   
    cos_similarities = cos_similarity(E.vectors, v)
    return np.argsort(-cos_similarities, axis=0)[:k, :]


def demo_similarity(words, embedding_model):
    embeddings = np.vstack([embedding_model.embed_word(word) for word in words])
    cos_similarities = cos_similarity(embeddings, embeddings)
    table = pd.DataFrame(cos_similarities, index=words, columns=words)
    print(table)

def demo_topk(words, k, embedding_model):
    embeddings = np.vstack([embedding_model.embed_word(word) for word in words])
    topk = topk_ids(embeddings, embedding_model, k)

    for i, word in enumerate(words):
        print(f'{word} top-{k} closest: {[embedding_model.word(index) for index in topk[:, i]]}')


def demo_analogy(a, b, c, k, embedding_model):
    emb = embedding_model.embed_word(a) - embedding_model.embed_word(b) + embedding_model.embed_word(c)
    emb = emb[np.newaxis, :]
    topk = topk_ids(emb, embedding_model, k)
    print(f'{a}-{b}+{c} top-{k} closest: {[embedding_model.word(index) for index in topk[:, 0]]}')
