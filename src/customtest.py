import pandas as pd
import numpy as np
# from src.data_util.fm_preprocess import FM_Preprocessing
from src.util.preprocessor import Preprocessor
import tqdm
import torch
# import copy

class Tester:

    def __init__(self, args, model, data: Preprocessor) -> None:

        self.args = args
        self.model = model

        self.train_df, self.test_org = data.train_df, data.train_org
        self.user_df, self.item_df = data.user_info, data.item_info
        self.le_dict = data.le_dict  # le is labelencoder
        self.user_embedding, self.item_embedding = data.user_embedding_df, data.item_embedding_df
        self.cat_cols, self.cont_cols = data.cat_columns, data.cont_columns   
        self.train_org = data.train_org

    # to make dataframe with given user_id
    def test_data_generator(self, user_id):
        item_ids = self.le_dict['item_id'].classes_
        user_ids = np.repeat(user_id, len(item_ids))
        
        test_df = pd.DataFrame({"user_id" : user_ids, "item_id" : item_ids})
        test_df = pd.merge(test_df, self.item_df, on='item_id', how='left')
        test_df = pd.merge(test_df, self.user_df, on='user_id', how='left')
        test_df = pd.merge(test_df.set_index('item_id'), self.item_embedding, on='item_id', how='left')
        test_df = pd.merge(test_df.set_index('user_id'), self.user_embedding, on='user_id', how='left')

        final_df = test_df

        for col in self.cat_cols: # 카테고리 column들을 차례로
            final_df[col] = self.le_dict[col].transform(final_df[col]) # 각 label encoder을 이용해 transform
        if self.args.embedding_type=='SVD': # embedding type이 SVD면 user_id와 item_id도 transform시켜줌
            final_df['user_id'] = self.le_dict['user_id'].transform(final_df['user_id'])
            final_df['item_id'] = self.le_dict['item_id'].transform(final_df['item_id'])

        return final_df

    def test(self):
        train_org = self.train_org.copy(deep=True)
        train_org['user_id'] = self.le_dict['user_id'].transform(train_org['user_id'])
        train_org['item_id'] = self.le_dict['item_id'].transform(train_org['item_id'])

        user_list = self.le_dict['user_id'].classes_
        self.model.eval()
        precisions, recalls, hit_rates, reciprocal_ranks, dcgs = [], [], [], [], []

        for customerid in tqdm.tqdm(user_list[:]):

            temp = self.test_data_generator(customerid)
            X_cat = temp[self.cat_cols].value
            X_cat = torch.tensor(X_cat, dtype=torch.int64)
            X_cont = temp[self.cont_cols].values
            X_cont = torch.tensor(X_cont, dtype=torch.float32)
    
            if self.args.model_type=='fm':
                emb_x = self.model.embedding(X_cat)
                result, _, _, _ = self.model.forward(X_cat, X_cont, emb_x)
            else:
                result = self.model.forward(X_cat, X_cont)
            
            topidx = torch.argsort(result, descending=True)[:]
            topidx = topidx.tolist()

            if customerid not in self.test_org['user_id'].unique():
                continue

            print("customer id: ",customerid, end=" ")
            ml = list(self.le_dict['item_id'].inverse_transform(temp['item_id'].unique()))
            ml = np.array(ml)
            ml = ml[topidx] # reorder movie_list
            cur_userslist = np.array(train_org[(train_org['user_id'])==self.le_dict['user_id'].transform([customerid])[0]]['item_id'].unique())
            
            # 여기 안본게 포함되어있을 수 있음 이거 처리해줘야함
            cur_userslist = self.le_dict['item_id'].inverse_transform(cur_userslist)
            
            # erase the things in ml that are in cur_userslist without changing the order
            real_rec = np.setdiff1d(ml,cur_userslist,assume_unique=True)
            
            print("top {} recommended product code: ".format(self.args.topk),real_rec[:self.args.topk])

            cur_user_test=np.array(self.test_org[(self.test_org['user_id'])==customerid])
            cur_user_test=cur_user_test[:,1]
            cur_user_test=np.unique(cur_user_test)
            cur_user_test=cur_user_test.tolist()

            if(len(cur_user_test)==0 or len(cur_user_test)<self.args.topk):
                continue
            # print("real product code: ",cur_user_test[:])
            # real_rec = real_rec.tolist()

            precision = self.get_precision(real_rec[:self.args.topk], cur_user_test)
            recall = self.get_recall(real_rec[:self.args.topk], cur_user_test)
            hit_rate = self.get_hit_rate(real_rec[:self.args.topk], cur_user_test)
            reciprocal_rank = self.get_reciprocal_rank(real_rec[:self.args.topk],cur_user_test)
            dcg = self.get_dcg(real_rec[:self.args.topk],cur_user_test)
            
            precisions.append(precision)
            recalls.append(recall)
            hit_rates.append(hit_rate)
            reciprocal_ranks.append(reciprocal_rank)
            dcgs.append(dcg)
  
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("hit rate: ", hit_rate)
            # print("reciprocal rank: ", reciprocal_rank)
            # print("dcg: ", dcg)
        
        print("average precision: ", np.mean(precisions))
        print("total user number: ", len(user_list))
        print("total item number: ", len(self.train_df['item_id'].unique()))
        metrics = {'precision' : np.mean(precisions), 
                   'recall' : np.mean(recalls),
                   'hit_rate' : np.mean(hit_rates),
                   'reciprocal_rank' : np.mean(reciprocal_ranks),
                   'dcg' : np.mean(dcgs)}

        return metrics
    

    def svdtest(self):
        train_org = self.train_org.copy(deep=True)
        train_org['user_id'] = self.le_dict['user_id'].transform(train_org['user_id'])
        train_org['item_id'] = self.le_dict['item_id'].transform(train_org['item_id'])

        user_list = self.le_dict['user_id'].classes_
        self.model.eval()
        precisions, recalls, hit_rates, reciprocal_ranks, dcgs = [], [], [], [], []
        
        for user_id in tqdm.tqdm(user_list[:]):

            temp = self.test_data_generator(user_id)
            X_cat = temp[self.cat_cols].values
            X_cat = torch.tensor(X_cat, dtype=torch.int64) # test_data_generator의 값 중 categorical column들의 값을 포함하는 텐서
            X_cont = temp[self.cont_cols].values
            X_cont = torch.tensor(X_cont, dtype=torch.float32) # test_data_generator의 값 중 continuous columns(embeddings?)을 포함하는 텐서

            svd_emb = X_cont[:, -self.args.num_eigenvector*2:]
            X_cont = X_cont[:, :-self.args.num_eigenvector*2]
            emb_x = self.model.embedding(X_cat)

            if self.args.model_type=='fm':
                result, _, _, _ = self.model.forward(X_cat, emb_x, svd_emb, X_cont)
            else: # model_type=='deepfm'
                result = self.model.forward(X_cat, emb_x, svd_emb, X_cont)
            
            topidx = torch.argsort(result,descending=True)[:]
            topidx = topidx.tolist()

            if user_id not in self.test_org['user_id'].unique():
                continue

            print("customer id: ", user_id, end=" ")
            ml = list(self.le_dict['item_id'].inverse_transform(temp['item_id'].unique()))
            ml = np.array(ml)
            ml = ml[topidx] # reorder movie_list
            cur_userslist = np.array(train_org[(train_org['user_id'])==self.le_dict['user_id'].transform([user_id])[0]]['item_id'].unique())
            
            # testing needs to be done with item_id that exists in train data
            cur_userslist = self.le_dict['item_id'].inverse_transform(cur_userslist)
            
            # erase the things in ml that are in cur_userslist without changing the order
            real_rec = np.setdiff1d(ml, cur_userslist, assume_unique=True)
            
            print("top {} recommended product code: ".format(self.args.topk), real_rec[:self.args.topk])

            cur_user_test = np.array(self.test_org[(self.test_org['user_id'])==user_id])
            cur_user_test = cur_user_test[:, 1]
            cur_user_test = np.unique(cur_user_test)
            cur_user_test = cur_user_test.tolist()

            if (len(cur_user_test)==0 or len(cur_user_test)<self.args.topk):
                continue
            print("real product code: ", cur_user_test[:])
            real_rec = real_rec.tolist()

            pred = real_rec[:self.args.topk]
            real = cur_user_test
            precision = self.get_precision(pred, real)
            recall = self.get_recall(pred, real)
            hit_rate = self.get_hit_rate(pred, real)
            reciprocal_rank = self.get_reciprocal_rank(pred, real)
            dcg = self.get_dcg(pred, real)

            precisions.append(precision)
            recalls.append(recall)
            hit_rates.append(hit_rate)
            reciprocal_ranks.append(reciprocal_rank)
            dcgs.append(dcg)
  
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("hit rate: ", hit_rate)
            # print("reciprocal rank: ", reciprocal_rank)
            # print("dcg: ", dcg)
        
        print("average precision: ", np.mean(precisions))
        print("total user number: ", len(user_list))
        print("total item number: ", len(self.train_df['item_id'].unique()))
        metrics = {'precision' : np.mean(precisions),
                   'recall' : np.mean(recalls),
                   'hit_rate' : np.mean(hit_rates),
                   'reciprocal_rank' : np.mean(reciprocal_ranks),
                   'dcg' : np.mean(dcgs)}

        return metrics
    
    # metric 함수
    def get_precision(self, pred, real):
        precision = len(set(pred).intersection(set(real)))/len(pred)
        return precision
    
    def get_recall(self, pred, real):
        recall = len(set(pred).intersection(set(real)))/len(real)
        return recall
    
    def get_hit_rate(self, pred, real):
        if len(set(pred).intersection(set(real)))>0:
            return 1
        else:
            return 0
    
    def get_reciprocal_rank(self, pred, real):
        for i in range(len(pred)):
            if pred[i] in real:
                return 1/(i+1)
        return 0
    
    def get_dcg(self, pred, real):
        dcg = 0
        for i in range(len(pred)):
            if pred[i] in real:
                dcg += 1/np.log2(i+2)
        return dcg
