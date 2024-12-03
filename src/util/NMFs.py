from sklearn.decomposition import NMF

class NMFs:

    def __init__(self, args):
        self.args = args
    
    def fit_nmf(self, x):
        nmf = NMF(n_components=16, random_state=42)
        W = nmf.fit_transform(x)
        H = nmf.components_
        return W, H.T
