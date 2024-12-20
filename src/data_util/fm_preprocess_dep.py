# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import tqdm
# import random
# from copy import deepcopy

# 사용된 것을 본 적이 없음???
# class FM_Preprocessing:

#     def __init__(self, args,df, target_col='target', num_epochs=10):
#         self.args = args
#         self.df = df
#         self.target_col = target_col
#         self.num_epochs = num_epochs
#         self.X_tensor, self.y_tensor, self.c_values_tensor, self.user_feature_tensor, self.item_feature_tensor, self.all_item_ids, self.num_features, self.dics = self.prepare_data()
#         if not isinstance(df, pd.DataFrame):
#             raise ValueError("The df parameter should be a pandas DataFrame.")
        
#         if target_col not in df.columns:
#             raise ValueError(f"The target column {target_col} is not in the DataFrame.")

#     def prepare_data(self):
#         X = self.df # temporary
#         X = X.drop(['target','c','user_id','movie_id'], axis=1, inplace=False)
#         y = self.df['target']
#         c = self.df['c']

#         # to unity dtypes to float
#         X = X.astype(float)

#         X_tensor = torch.tensor(X.values, dtype=torch.int64)
#         y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1)

#         c_values_tensor = torch.tensor(c, dtype=torch.float32)
#         c_values_tensor = torch.where(c_values_tensor < 1, c_values_tensor*1 , c_values_tensor)

#         # want to make user_id and product_id mapping dictionary

#         user_feature_tensor = 1
#         item_feature_tensor = 1
#         all_item_ids = 1
#         num_features = X.shape[1]
#         dics = 1
        
#         return X_tensor, y_tensor, c_values_tensor, user_feature_tensor, item_feature_tensor, all_item_ids, num_features,dics