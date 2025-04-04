import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from copy import deepcopy
from typing import Tuple

from src.util.negativesampler import NegativeSampler
from src.util.embed_SVD import embed_SVD
from src.util.embed_NMF import embed_NMF
from src.util.embed_SparseSVD import embed_SparseSVD


class Preprocessor:
    """
    Preprocessor class for preprocessing the input data
    """
    def __init__(self, args, train_df, test_df, user_info, item_info, 
                 ui_matrix, cat_columns, cont_columns):
        """
        Constructor for Preprocessor class
        """
        self.args = args
        self.train_org = train_df.copy(deep=True)
        self.train_df = train_df
        self.test_df = test_df
        self.item_info = item_info
        self.user_info = user_info
        self.ui_matrix = ui_matrix
        self.preprocess(train_df, item_info, user_info)
        self.label_encode(cat_columns)
        self.alter_dfs(cat_columns, cont_columns)

        
    def get_original_train(self) -> pd.DataFrame:
        return self.train_org

    def preprocess(self, train_df, item_info, user_info) -> None:
        """
        Method to preprocess the input data
        :return: Preprocessed data
        """ 
        # Negative Sampling
        ns = NegativeSampler(self.args, train_df, item_info, user_info)
        ns_sampled_df = ns.negativesample(self.args.isuniform)

        self.target = ns_sampled_df['target'].to_numpy()
        self.c = ns_sampled_df['c'].to_numpy()
        ns_sampled_df.drop(['target', 'c'], axis=1, inplace= True)

        # merge item_info and user_info => 나중에 merge하는 작업은 밑에 있는 embedding merge에서 하는걸로 처리해주기
        ns_sampled_df = ns_sampled_df.merge(item_info, on='item_id', how='left')
        ns_sampled_df = ns_sampled_df.merge(user_info, on='user_id', how='left')
        
        # ui_matrix를 user_embedding, item_embedding으로 행렬 분해
        if self.args.embedding_type=='original':
            self.train_df = ns_sampled_df.reset_index(drop=True)
        else:
            if self.args.embedding_type=='SVD':
                user_embedding, item_embedding  = embed_SVD(self.args).fit_truncatedSVD(self.ui_matrix)
            elif self.args.embedding_type=='SparseSVD':
                user_embedding, item_embedding = embed_SparseSVD(self.args).fit_sparse_svd(self.ui_matrix)
            else:
                user_embedding, item_embedding = embed_NMF(self.args).fit_nmf(self.ui_matrix)
            self.train_df, self.user_embedding_df, self.item_embedding_df = self.merge_embedding(user_embedding, item_embedding, ns_sampled_df)
    
    def merge_embedding(self, user_embedding, item_embedding, ns_sampled_df):
        user_ids = pd.Series(sorted(ns_sampled_df['user_id'].unique()))
        item_ids = pd.Series(sorted(ns_sampled_df['item_id'].unique()))
        
        user_embedding_df = pd.concat([user_ids, pd.DataFrame(user_embedding)], axis=1)
        item_embedding_df = pd.concat([item_ids, pd.DataFrame(item_embedding)], axis=1)

        user_embedding_df.columns = ['user_id'] + [f'user_embedding_{i}' for i in range(user_embedding.shape[1])]
        item_embedding_df.columns = ['item_id'] + [f'item_embedding_{i}' for i in range(item_embedding.shape[1])]

        movie_emb_included_df = pd.concat([ns_sampled_df.reset_index(drop=True), 
                                           item_embedding_df.set_index('item_id').reindex(ns_sampled_df['item_id'].values).reset_index(drop=True)], 
                                           axis=1)
        user_emb_included_df = pd.concat([movie_emb_included_df.reset_index(drop=True), 
                                          user_embedding_df.set_index('user_id').reindex(movie_emb_included_df['user_id'].values).reset_index(drop=True)], 
                                          axis=1)

        return user_emb_included_df, user_embedding_df, item_embedding_df

    
    def label_encode(self, cat_columns):
        # label_encoders is a dictionary for label_encoder, holds label encoder for each categorical column
        self.le_dict = {}
        
        # when we are using original embedding, we need to encode user_id and item_id
        if self.args.embedding_type=='original':
            for col in cat_columns:
                le = LabelEncoder()
                self.train_df[col] = le.fit_transform(self.train_df[col])
                self.le_dict[col] = le
        # when we are using SVD, we don't need to embed user_id and item_id
        else:
            for col in cat_columns:
                le = LabelEncoder()
                if col=='user_id' or col=='item_id':
                    le.fit(self.train_df[col])
                else:
                    self.train_df[col] = le.fit_transform(self.train_df[col])
                self.le_dict[col] = le
            
    
    def alter_dfs(self, cat_columns, cont_columns):
        self.cont_train_df = self.train_df.drop(cat_columns, axis=1)
        if self.args.embedding_type!='original':
            cat_columns.remove('user_id')
            cat_columns.remove('item_id')
            cont_columns = cont_columns + self.user_embedding_df.columns.tolist()[1:] + self.item_embedding_df.columns.tolist()[1:]
        
        self.cat_columns = cat_columns
        self.cont_columns = cont_columns
        self.cat_train_df = self.train_df[cat_columns].to_numpy()[:].astype('int')
        self.cont_train_df = self.cont_train_df[cont_columns].to_numpy()[:].astype('float32')
        
        self.args.cont_dims = len(cont_columns)
        self.uidf = self.train_df[['user_id', 'item_id']]
        self.field_dims = np.max(self.cat_train_df, axis=0) + 1