import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from copy import deepcopy
from typing import Tuple

from src.util.negativesampler import NegativeSampler
from src.util.SVD import SVD


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
        self.cat_columns = cat_columns
        self.cont_columns = cont_columns
        self.preprocess()
    
    def get_original_train(self) -> pd.DataFrame:
        return self.train_org

    def get_user_item_info(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.user_info, self.item_info # returns two pd.DataFrame

    def get_catcont_train(self) -> Tuple[np.ndarray, np.ndarray]: # get categorical and continuous train data
        return self.cat_train_df_temp, self.cont_train_df_temp
    
    def get_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]: # get train and test data
        return self.train_df_temp, self.test_df
    
    def get_column_info(self) -> Tuple[list, list]: # get column information
        return self.cat_columns_temp, self.cont_columns_temp
    
    def get_embedding(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.user_embedding_df, self.item_embedding_df
    
    def get_le_dict(self) -> dict:
        return self.le_dict
    
    def get_field_dims(self) -> np.ndarray:
        return self.field_dims

    def get_target_c(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.target, self.c

    def preprocess(self):
        """
        Method to preprocess the input data
        :param data: Input data
        :return: Preprocessed data
        """ 
        # Negative Sampling
        ns = NegativeSampler(self.args, self.train_df, self.item_info, self.user_info)
        ns_sampled_df = ns.negativesample(self.args.isuniform)
        self.target = ns_sampled_df['target'].to_numpy()
        self.c = ns_sampled_df['c'].to_numpy()
        ns_sampled_df.drop(['target', 'c'],axis=1,inplace= True)

        # merge item_info and user_info => 나중에 merge하는 작업은 밑에 있는 embedding merge에서 하는걸로 처리해주기
        ns_sampled_df = ns_sampled_df.merge(self.item_info, on='item_id', how='left')
        self.ns_sampled_df = ns_sampled_df.merge(self.user_info, on='user_id', how='left')
        # ui_matrix를 user_embedding, item_embedding으로 SVD를 이용하여 행렬 분해
        # 특이값을 제외하고 U랑 V는 달라질 수 있는데 영향은?
        self.user_embedding, self.item_embedding = SVD(self.args).fit_truncatedSVD(self.ui_matrix)
        self.train_df, self.user_embedding_df, self.item_embedding_df = self.merge_embedding(self.user_embedding, self.item_embedding)

    
    def merge_embedding(self, user_embedding, item_embedding):
        # from trainingdf if user_id is 1, then user_embedding[0] is the embedding
        # from trainingdf if user_id is 1, then movie_embedding[0] is the embedding
        """
        user_embedding and movie_embedding are both numpy arrays
        user_embedding.shape[0] is the number of users
        """
        user_embedding_df = pd.DataFrame()
        item_embedding_df = pd.DataFrame()

        user_embedding_df['user_id'] = sorted(self.ns_sampled_df['user_id'].unique())
        item_embedding_df['item_id'] = sorted(self.ns_sampled_df['item_id'].unique())

        user_embedding_columns = []
        item_embedding_columns = []

        for i in range(user_embedding.shape[1]):
            user_embedding_columns.append('user_embedding_'+str(i))
        for i in range(item_embedding.shape[1]):
            item_embedding_columns.append('item_embedding_'+str(i))

        ue_df = pd.DataFrame(user_embedding, columns=user_embedding_columns)
        ie_df = pd.DataFrame(item_embedding, columns=item_embedding_columns)

        user_embedding_df = pd.concat([user_embedding_df, ue_df], axis=1)
        item_embedding_df = pd.concat([item_embedding_df, ie_df], axis=1)

        movie_emb_included_df = pd.concat([self.ns_sampled_df.reset_index(drop=True), 
                                           item_embedding_df.set_index('item_id').reindex(self.ns_sampled_df['item_id'].values).reset_index(drop=True)], 
                                           axis=1)
        user_emb_included_df = pd.concat([movie_emb_included_df.reset_index(drop=True), 
                                          user_embedding_df.set_index('user_id').reindex(movie_emb_included_df['user_id'].values).reset_index(drop=True)], 
                                          axis=1)

        return user_emb_included_df, user_embedding_df, item_embedding_df

    
    def label_encode(self):
        self.cont_train_df = self.train_df.drop(self.cat_columns, axis=1)
        # deep copy
        train_df = self.train_df.copy(deep=True)

        cont_columns = deepcopy(self.cont_columns)
        cat_columns = deepcopy(self.cat_columns)
        
        # label_encoders is a dictionary for label_encoder, holds label encoder for each categorical column
        self.le_dict = {}
        # when we use SVD, we don't need to embedd user_id and item_id
        if self.args.embedding_type=='SVD':
            # cat_columns 
            #  ['user_id', 'item_id', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
            #   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
            #   'Thriller', 'War', 'Western', 'age', 'gender', 'occupation']
            for col in cat_columns:
                le = LabelEncoder()
                if col=='user_id' or col=='item_id':
                    le.fit(train_df[col])
                else:
                    train_df[col] = le.fit_transform(train_df[col])
                self.le_dict[col] = le
    
            cat_train_df = train_df[cat_columns].drop(['user_id','item_id'], axis=1).to_numpy()[:].astype('int')
            # array([[ 0,  0,  0, ...,  2,  1, 19],
            #        [ 0,  1,  1, ...,  2,  1, 19],
            #        [ 0,  0,  0, ...,  2,  1, 19],
            #         ...,
            #        [ 0,  0,  0, ...,  1,  1, 18],
            #        [ 0,  0,  0, ...,  1,  1, 18],
            #        [ 0,  1,  0, ...,  1,  1, 18]]
            cont_columns = cont_columns + self.user_embedding_df.columns.tolist() + self.item_embedding_df.columns.tolist()
            # ['user_id', 'user_embedding_0', 'user_embedding_1', 'user_embedding_2', 'user_embedding_3', 'user_embedding_4', 
            #  'user_embedding_5', 'user_embedding_6', 'user_embedding_7', 'user_embedding_8', 'user_embedding_9', 'user_embedding_10', 
            #  'user_embedding_11', 'user_embedding_12', 'user_embedding_13', 'user_embedding_14', 'user_embedding_15', 
            #  'item_id', 'item_embedding_0', 'item_embedding_1', 'item_embedding_2', 'item_embedding_3', 'item_embedding_4', 'item_embedding_5', 'item_embedding_6', 
            #  'item_embedding_7', 'item_embedding_8', 'item_embedding_9', 'item_embedding_10', 'item_embedding_11', 'item_embedding_12', 'item_embedding_13', 
            #  'item_embedding_14', 'item_embedding_15']
            
            # user_id, item_id 삭제
            cont_columns.remove('user_id')
            cont_columns.remove('item_id')
            # ['user_embedding_0', 'user_embedding_1', 'user_embedding_2', 'user_embedding_3', 'user_embedding_4', 'user_embedding_5', 'user_embedding_6', 'user_embedding_7', 
            #  'user_embedding_8', 'user_embedding_9', 'user_embedding_10', 'user_embedding_11', 'user_embedding_12', 'user_embedding_13', 'user_embedding_14', 'user_embedding_15', 
            #  'item_embedding_0', 'item_embedding_1', 'item_embedding_2', 'item_embedding_3', 'item_embedding_4', 'item_embedding_5', 'item_embedding_6', 'item_embedding_7', 
            #  'item_embedding_8', 'item_embedding_9', 'item_embedding_10', 'item_embedding_11', 'item_embedding_12', 'item_embedding_13', 'item_embedding_14', 'item_embedding_15']

            cont_train_df = self.cont_train_df[cont_columns]
              #         user_embedding_0  user_embedding_1  user_embedding_2  user_embedding_3  user_embedding_4  ...  item_embedding_11  item_embedding_12  item_embedding_13  item_embedding_14  item_embedding_15
              # 0               4.856453         -0.399200          0.575379         -1.154136         -0.699106  ...          -0.001916          -0.041549           0.068542          -0.018827          -0.016799
              # 1               4.856453         -0.399200          0.575379         -1.154136         -0.699106  ...          -0.004488          -0.009219           0.010461           0.023436           0.033961
              # 2               4.856453         -0.399200          0.575379         -1.154136         -0.699106  ...           0.001123          -0.056571           0.006716          -0.026319           0.047488
              # 3               4.856453         -0.399200          0.575379         -1.154136         -0.699106  ...           0.054508           0.057809          -0.063818           0.027591          -0.070931
              # 4               4.856453         -0.399200          0.575379         -1.154136         -0.699106  ...           0.016208           0.040225          -0.000089           0.051143          -0.041623
              # [374829 rows x 32 columns]
            self.args.cont_dims = len(cont_columns)
            cat_columns.remove('user_id')
            cat_columns.remove('item_id')
            
        # when we use original embedding, we need to encode user_id and item_id
        else:
            for col in cat_columns:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col])
                self.le_dict[col] = le
            cat_train_df = train_df[cat_columns].to_numpy()[:].astype('int')
            cont_train_df = self.cont_train_df[cont_columns]
            self.args.cont_dims = len(cont_columns)

        self.cat_columns_temp = cat_columns
        self.cont_columns_temp = cont_columns
        self.uidf = train_df[['user_id', 'item_id']]
        self.cat_train_df_temp = cat_train_df
        self.cont_train_df_temp = cont_train_df.to_numpy()[:].astype('float32')
        
        self.field_dims = np.max(self.cat_train_df_temp, axis=0) + 1
        self.train_df_temp = train_df.copy(deep=True)
