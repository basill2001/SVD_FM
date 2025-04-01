from sklearn.decomposition import MiniBatchSparsePCA

class embed_SparseSVD:

    def __init__(self, args) -> None:
        self.args = args
        pass

    def fit_sparse_svd(self, x):
        """
        sparse matrix(x)와 number of singular values(k)를 입력받으면 
        SVD 행렬분해 수행
        """
        print("pca 1 start")
        pca1 = MiniBatchSparsePCA(n_components=self.args.num_eigenvector, 
                                  alpha=self.args.sparsity, batch_size=100)
        pca1.fit(x@x.T)
        u = pca1.components_
        print("pca 2 start")
        pca2 = MiniBatchSparsePCA(n_components=self.args.num_eigenvector, 
                                  alpha=self.args.sparsity, batch_size=100)
        pca2.fit(x.T@x)
        v = pca2.components_

        return u.T, v.T

    # def fit_truncatedSVD(self, x):
    #     truncsvd = TruncatedSVD(n_components=self.args.num_eigenvector)
    #     u = truncsvd.fit_transform(x)
    #     v = truncsvd.components_

    #     return u, v.T