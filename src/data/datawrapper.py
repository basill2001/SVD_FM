from src.data.movielens100k import Movielens100k
from src.data.movielens1m import Movielens1m
from src.data.shoppingdata import ShoppingData
from src.data.movielens10m import Movielens10m
from src.data.frappe import Frappe
from src.data.goodbook import GoodBook
import tqdm

class DataWrapper:

    def __init__(self, args) -> None:
        pass

        # datatype에 따른 데이터를 self.data에 저장
        if args.datatype=="ml100k": # 수정필요
            self.data = Movielens100k('dataset/ml-100k','u.data', args.fold)
        elif args.datatype=="ml1m": # 수정필요
            self.data = Movielens1m(args)
        elif args.datatype=="shopping": # 수정필요
            self.data = ShoppingData(args)
        elif args.datatype=="ml10m": # 데이터 사용불가 (현재로써는. )
            self.data = Movielens10m(args)
        elif args.datatype=="frappe":
            self.data = Frappe(args) # 수정완료
        elif args.datatype=="goodbook":
            self.data = GoodBook(args)
        else:
            raise NotImplementedError

        # 각 data마다 만들어진 class에 있는 data_getter 사용
        self.train, self.test, self.item_info, self.user_info = self.data.data_getter()



    def get_data(self):
        self.ui_matrix = self.get_user_item_matrix()

        return self.train, self.test, self.item_info, self.user_info, self.ui_matrix
    
    
    def get_user_item_matrix(self):

        #get useritem matrix
        train = self.train
        # 행이 user 열이 item인 관계 matrix 생성
        useritem_matrix = train.pivot_table(index='user_id',columns='item_id',values='rating')
        useritem_matrix = useritem_matrix.fillna(0)
        useritem_matrix =useritem_matrix.astype(float)
        # 구매했을 경우 1로 설정
        useritem_matrix[useritem_matrix >= 3] = 1
        useritem_matrix[useritem_matrix < 3]  = 0
        useritem_matrix = useritem_matrix.to_numpy()
        # x dtype to float``
        useritem_matrix=useritem_matrix.astype(float)  

        return useritem_matrix
    


    def get_col_type(self):
        
        # 범주형, 연속형을 나누어서 저장
        cat_cols = []
        cont_cols = []
        cat_cols.append('user_id')
        cat_cols.append('item_id')

        for col in self.item_info.columns:
            if col=='item_id':
                continue
            if self.item_info[col].dtype=='object' :
                cat_cols.append(col)
            elif self.item_info[col].dtype=='int64' :
                cat_cols.append(col)
            elif self.item_info[col].dtype=='float64' :
                cont_cols.append(col)

        for col in self.user_info.columns:
            if col=='user_id':
                continue

            if self.user_info[col].dtype=='object' :
                cat_cols.append(col)
            elif self.user_info[col].dtype=='int64' :
                cat_cols.append(col)
            elif self.user_info[col].dtype=='float64' :
                cont_cols.append(col)
        return cat_cols, cont_cols