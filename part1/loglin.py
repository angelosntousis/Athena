from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm.auto import tqdm

def load(path):
    """
    Load a logistic model from {path}.pickle and return it.
    """
    with open(f"{path}.pickle", 'rb') as f:
        logistic_model = pickle.load(f)
    return logistic_model

def encode(texts, encoder: 'Encoder'):
    """
    Encodes texts as a sparse matrix (one row per text).
    Encoders from features.py are applied to the whole batch at once, which avoids building
    one dense vocabulary-sized vector per text; other encoders (such as clslib.TokenIndicator) text by text.
    """
    if hasattr(encoder, "transform"):
        return csr_matrix(encoder.transform(texts))
    return csr_matrix([encoder(text) for text in texts])

def fit(training_set, encoder: 'Encoder', save_to=None, max_iter=100, n_jobs=-1, C=1.0, l1_ratio=0.0, tol=1e-4, random_state=None):
    """
    Constructs, trains and returns a softmax regressor.

    training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
    encoder: an instance of the Encoder class (such as TokenIndicator)
    save_to: (str) if given, save the model to {save_to}.pickle
    max_iter: maximum number of passes (epochs) over the training data
    C: inverse regularisation strength (smaller C, stronger regularisation; float('inf') for none)
    l1_ratio: 0.0 for an L2 penalty, 1.0 for L1 (values in between mix the two)
    tol: training stops early once the weights change less than this between epochs
    random_state: seed for the order in which saga visits the training examples
    """
    # Convert the training set to a sparse matrix, which makes training more efficient
    X_train_sparse = encode([text for text, label in training_set], encoder)
    training_labels = [label for text, label in training_set]

    # Train the logistic model
    logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs, C=C, l1_ratio=l1_ratio,
                                        tol=tol, random_state=random_state) #sparse matrix frinedly solver
    logistic_model.fit(X_train_sparse, training_labels)

    if save_to is not None:        
        with open(f"{save_to}.pickle", 'wb') as f:
            pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return logistic_model        

def predict(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    X_val_sparse = encode(texts, encoder)
    predictions = model.predict(X_val_sparse)
    return predictions

def predict_probs(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    X_val_sparse = encode(texts, encoder)
    probs = model.predict_proba(X_val_sparse)
    return probs