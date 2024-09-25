from copy import deepcopy
import numpy as np
import pandas as pd
import tqdm

class NegativeSampler:
    #  takes input of original dataframe and movie info
    #  make a function that returns negative sampled data

    def __init__(self, args, original_df, item_info, user_info) -> None:
        self.args=args
        self.original_df = original_df
        self.seed=np.random.seed(args.seed)
        self.original_df.drop(columns=['timestamp','rating'], axis=1, inplace=True)
        self.original_df['target']=1
        self.original_df['c']=1
        self.item_info=item_info
        self.user_info=user_info
        pass

    # function that calculates the c value for each customer and product
    # higher beta -> more weight on product frequency
    # higher alpha -> more weight on customer frequency

    def get_c(self, df, uu_sum, ii_sum, alpha=.5, beta=.5, gamma=.5, c_zero=10):
        c_zero = self.args.c_zeros
        UF = np.array(df["user_frequency"].astype("float"), dtype=float)
        UF /= uu_sum # 전체 중 비율 (개별/ Total)
        IF = np.array(df["item_frequency"].astype("float"), dtype=float)
        IF /= ii_sum # 전체 중 비율 (개별/Total)
        Fs = alpha * beta * IF * UF
        Fs_gamma = Fs ** gamma
        c = Fs_gamma / np.sum(Fs_gamma)
        c = c_zero * c / np.max(c)
        c_appended_df = deepcopy(df)
        c_appended_df['c'] = c

        return c_appended_df
    
    def negativesample(self, isuniform=False):

        unique_customers = self.original_df['user_id'].unique()
        df=self.original_df
        not_purchased_df = pd.DataFrame()
        ns_df_list = []
        df['user_frequency'] = df.groupby('user_id')['user_id'].transform('count')
        df['item_frequency'] = df.groupby('item_id')['item_id'].transform('count')
        #multiprocess

        print("Negative Sampling Started")

        for customer in tqdm.tqdm(unique_customers[:]):
            #unique_products = df['item_id'].unique()


            customer_frequency = df[df['user_id'] == customer]['user_frequency'].iloc[0]
            purchased_products = df[df['user_id'] == customer]['item_id'].unique()

            #customer_birth_category = df[df['user_id'] == customer]['BIRTH_YEAR'].iloc[0]
            #customer_gender_category = df[df['user_id'] == customer]['GENDER'].iloc[0]

            not_purchased_df_all = df[~df['item_id'].isin(purchased_products)]
            # user가 구매하지 않은 item들의 코드
            not_purchased_codes = not_purchased_df_all['item_id'].unique()
            # 구매하지 않은 item들 중 n개(ratio_neagtive에 따름) 선택
            negative_sample_products = np.random.choice(not_purchased_codes, int(len(not_purchased_codes) *self.args.ratio_negative), replace=False)
            
            # negative_sample_products의 item id와 frequency가 있는 df
            ns_test = df[df['item_id'].isin(negative_sample_products)][['item_id','item_frequency']]
            ns_test = ns_test.drop_duplicates(subset=['item_id'], keep='first', inplace=False)

            # negative_sample_products의 item_id, user_id, user_frq, item_frq가 있는 df
            ns_df = pd.DataFrame()
            ns_df['item_id'] = negative_sample_products
            ns_df=ns_df.assign(user_id = customer)
            #ns_df['AUTH_CUSTOMER_ID'] = customer
            #ns_df=ns_df.assign(BIRTH_YEAR = customer_birth_category)
            #ns_df=ns_df.assign(GENDER= customer_gender_category)
            ns_df=ns_df.assign(user_frequency = customer_frequency)
            # mergge
            ns_df=pd.merge(ns_df, ns_test, on='item_id', how='left')


            # ns_df=ns_df.join(ns_test, on='item_id')
            # not_purchased_df=pd.concat([not_purchased_df,ns_df],axis=0, ignore_index=True)
            ns_df_list += [ns_df]
        not_purchased_df = pd.concat(objs=ns_df_list, axis=0, ignore_index=True)
        
        #change column order
        not_purchased_df = not_purchased_df[['user_id','item_id','user_frequency','item_frequency']]
        # column of the not_purchased_df is 'user_id','item_id','user_frequency','item_frequency'

        not_purchased_df['target'] = 0

        # item마다 몇 번 구매되었는지
        mm = self.item_info['item_id'].map(self.original_df['item_id'].value_counts())
        # change nan to 0
        mm.fillna(0,inplace=True)
        # 총 구매횟수
        mm_sum = np.sum(mm.tolist())

        # user마다 몇 번 구매했는지
        uu=self.user_info['user_id'].map(self.original_df['user_id'].value_counts())
        # change nan to 0
        uu.fillna(0, inplace=True)
        # 총 구매횟수
        uu_sum=np.sum(uu.tolist())

        
        if isuniform:
            not_purchased_df['c'] = 1
        else:
            # user_frq, item_frq 이용
            not_purchased_df=self.get_c(not_purchased_df, uu_sum=uu_sum, ii_sum=mm_sum)

        # not_purchased_df.set_index('user_id',inplace=True)


        # original과 not_purchased_df 합쳐서 return
        to_return = pd.concat([self.original_df, not_purchased_df], axis=0, ignore_index=True)
        to_return.drop(['user_frequency','item_frequency'], axis=1, inplace=True)
        print("Negative Sampling Finished")
        return to_return