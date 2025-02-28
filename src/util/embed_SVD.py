from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import math

class embed_SVD:

    def __init__(self, args) -> None:
        self.args = args
        pass

    def fit_svd(self, x):
        """
        sparse matrix(x)와 number of singular values(k)를 입력받으면 
        SVD 행렬분해 수행
        """
        u, s, vt = svds(x, k=self.args.num_eigenvector)
        return u, vt.T

    def fit_truncatedSVD(self, x):
        truncsvd = TruncatedSVD(n_components=self.args.num_eigenvector)
        u = truncsvd.fit_transform(x)
        v = truncsvd.components_
        exp_var = sum(truncsvd.explained_variance_ratio_)
        full_var = truncsvd.full_variance_
        const_err = math.sqrt(full_var*(1-exp_var))

        return u, v.T, exp_var, const_err 